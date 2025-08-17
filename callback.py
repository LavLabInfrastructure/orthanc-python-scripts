"""Defines OrthancCallbackHandler which handles callbacks for the Orthanc plugin."""

import io
import json
import threading
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor

import pydicom
import pydicom.errors
from pynetdicom import AE, StoragePresentationContexts
from pynetdicom.association import Association

import orthanc  # type: ignore # pylint: disable=import-error
from config import Config
from processor import DicomProcessor
from excel import ExcelClient


# pylint: disable=unused-argument
class OrthancCallbackHandler:
    """Handles callbacks for the Orthanc plugin."""

    def __init__(
        self, config: Config, dicom_processor: DicomProcessor, excel_client: ExcelClient
    ):
        """Initialize the object with the given configuration, DICOM processor, and Excel client."""
        orthanc_config = config.get_orthanc_config()
        self.xnat_ip = orthanc_config["xnat_ip"]
        self.xnat_port = orthanc_config["xnat_port"]
        self.ae_title = orthanc_config["ae_title"]
        self.xnat_ae_title = orthanc_config["xnat_ae_title"]
        self.dicom_processor = dicom_processor
        self.excel_client = excel_client

        # Reusable, bounded worker pool to avoid spawning unbounded threads
        max_workers = int(orthanc_config.get("max_workers", 8) or 8)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Ensure only one /sync-all runs at a time
        self._sync_all_lock = threading.Lock()

    def send_to_xnat(self, ds: pydicom.Dataset) -> None:
        """Send the DICOM dataset to XNAT using C-STORE."""
        try:
            ae = AE(ae_title=self.ae_title)
            ae.requested_contexts = StoragePresentationContexts
            assoc: Association = ae.associate(
                self.xnat_ip, self.xnat_port, ae_title=self.xnat_ae_title
            )

            if assoc.is_established:
                status = assoc.send_c_store(ds)
                assoc.release()
                if status:
                    orthanc.LogInfo("Successfully sent DICOM file to XNAT")
                else:
                    orthanc.LogError("Failed to send DICOM file to XNAT")
            else:
                orthanc.LogError("Failed to associate with XNAT")
        except Exception as exc:  # pylint: disable=broad-except
            orthanc.LogError(f"Exception during C-STORE: {exc}")

    @staticmethod
    def _extract_patient_ids(ds: pydicom.Dataset) -> List[str]:
        """Extract possible alternate patient identifiers from standard fields.
        Returns a list of candidate IDs (may be empty)."""
        candidates: List[str] = []
        # Primary ID
        primary = ds.get("PatientID")
        if primary:
            try:
                candidates.append(str(primary))
            except Exception:  # pylint: disable=broad-except
                pass

        # OtherPatientIDs might be backslash-separated string or a sequence-like
        other_ids = ds.get("OtherPatientIDs")
        if other_ids:
            try:
                # If iterable (list/tuple), extend; else split string by '\\'
                if isinstance(other_ids, (list, tuple)):
                    candidates.extend(str(v) for v in other_ids if v)
                else:
                    for v in str(other_ids).split("\\"):
                        v = v.strip()
                        if v:
                            candidates.append(v)
            except Exception:  # pylint: disable=broad-except
                pass

        # OtherPatientIDsSequence
        seq = ds.get("OtherPatientIDsSequence")
        if seq:
            try:
                for item in seq:
                    for k in ("PatientID", "ID"):
                        val = item.get(k)
                        if val:
                            candidates.append(str(val))
            except Exception:  # pylint: disable=broad-except
                pass

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def process_instance(self, instance: Union[orthanc.DicomInstance, str]) -> None:
        """Process a new DICOM instance by re-identifying and sending it to XNAT."""
        try:
            if isinstance(instance, str):
                dcm_bytes = orthanc.GetDicomForInstance(instance)
            else:
                dcm_bytes = instance.GetInstanceData()
            if not dcm_bytes:
                orthanc.LogError(f"Could not get DICOM bytes for instance {instance}")
                return

            try:
                ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
                ds.filename = None
            except pydicom.errors.InvalidDicomError:
                orthanc.LogError(f"Invalid DICOM instance: {instance}")
                return

            sheet = self.dicom_processor.match_dicom_to_sheet(ds)
            formatting = sheet.get("format", "{}")
            # Build primary key and alternates from DICOM
            all_ids = self._extract_patient_ids(ds)
            if not all_ids:
                orthanc.LogWarning(
                    "No patient identifiers found in DICOM; using PatientID as-is"
                )
                all_ids = [str(ds.get("PatientID", ""))]
            key = (
                formatting.format(all_ids[0])
                if all_ids[0]
                else formatting.format(ds.get("PatientID", ""))
            )
            alt_keys = [formatting.format(i) for i in all_ids[1:]]

            new_patient_id = self.excel_client.get_patient_id(key, sheet, alt_keys)
            if not new_patient_id:
                new_patient_id = key
            if key == new_patient_id:
                orthanc.LogWarning(
                    f"No mapping found for PatientID: {ds.get('PatientID')}"
                )

            deidentified_ds = self.dicom_processor.deidentify_dicom(ds, new_patient_id)
            orthanc.LogInfo(
                f"Re-identified DICOM with new patient ID: {new_patient_id}"
            )
            self.send_to_xnat(deidentified_ds)
        except Exception as exc:  # pylint: disable=broad-except
            orthanc.LogError(
                f"Unhandled exception while processing instance {instance}: {exc}"
            )

    def on_stored_instance(self, instance_id: str, *args, **kwargs) -> None:
        """Callback function triggered when a new instance is stored in Orthanc."""
        try:
            orthanc.LogInfo(f"Queueing new instance: {instance_id}")
            self.executor.submit(self.process_instance, instance_id)
        except Exception as exc:  # pylint: disable=broad-except
            orthanc.LogError(f"Failed to queue instance {instance_id}: {exc}")

    def on_series_sync_call(self, output: orthanc.RestOutput, uri, **request):
        """Rest API callback function to sync a series to XNAT."""
        try:
            series_id = request.get("get", {}).get("id")
            if series_id is None:
                return output.SendHttpStatusCode(400)
            try:
                series_response: dict = json.loads(
                    orthanc.RestApiGet(f"/series/{series_id}")
                )
            except orthanc.OrthancException:
                return output.SendHttpStatusCode(404)
            except json.JSONDecodeError:
                return output.SendHttpStatusCode(500)

            instances = series_response.get("instances", [])
            for inst in instances:
                self.executor.submit(self.process_instance, inst)
            return output.SendHttpStatusCode(200)
        except Exception:  # pylint: disable=broad-except
            return output.SendHttpStatusCode(500)

    def on_all_sync_call(self, output, uri, **request):
        # If a sync-all is already in progress, reject with 409 Conflict
        if not self._sync_all_lock.acquire(blocking=False):
            orthanc.LogWarning(
                "/sync-all is already running; rejecting concurrent request"
            )
            return output.SendHttpStatusCode(409)
        try:
            instances = json.loads(orthanc.RestApiGet("/instances"))

            def _worker(ids: List[str]):
                try:
                    for inst in ids:
                        self.executor.submit(self.process_instance, inst)
                finally:
                    # Always release the lock when done scheduling
                    self._sync_all_lock.release()

            # Schedule in background so the REST call returns immediately
            threading.Thread(target=_worker, args=(instances,), daemon=True).start()
            return output.SendHttpStatusCode(200)
        except Exception:  # pylint: disable=broad-except
            # On error, ensure we release the lock
            try:
                self._sync_all_lock.release()
            except Exception:
                pass
            return output.SendHttpStatusCode(500)
