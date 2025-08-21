"""Defines OrthancCallbackHandler which handles callbacks for the Orthanc plugin."""

import io
import json
import threading
import time
import warnings
import re
import ast
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence

import pydicom
import pydicom.errors
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
from pynetdicom import AE, StoragePresentationContexts, build_context
from pynetdicom.association import Association

import orthanc  # type: ignore # pylint: disable=import-error
from config import Config
from processor import DicomProcessor
from excel import ExcelClient

# Reduce noisy warnings that cannot be prevented at read time
warnings.filterwarnings(
    "ignore", message=r".*unknown escape sequence.*", module="pydicom.charset"
)
# Also suppress common VR warnings during read; values will be sanitized afterward
warnings.filterwarnings(
    "ignore", message=r".*Invalid value for VR IS.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=r".*value length.*allowed for VR IS.*", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=r".*value length.*VR\.SH.*", category=UserWarning
)


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
        # Bound the number of queued tasks to protect memory over multi-day runs
        max_queue = int(orthanc_config.get("max_queue", 1000) or 1000)
        self._submit_sema = threading.Semaphore(max_queue)
        # Ensure only one /sync-all runs at a time
        self._sync_all_lock = threading.Lock()
        # Pending queue: map of ID -> set(instance_id) awaiting discovered mapping
        self._pending_lock = threading.RLock()
        self._pending_by_id: dict[str, set[str]] = {}
        # Cap to protect memory over multi-day runs
        self._pending_max_ids = int(
            orthanc_config.get("pending_max_ids", 10000) or 10000
        )
        # Per-instance retry cap when re-queuing due to missing mapping
        self._pending_retries: dict[str, int] = {}
        self._pending_retries_max = int(
            orthanc_config.get("pending_max_retries", 5) or 5
        )

    @staticmethod
    def _ensure_uncompressed(ds: pydicom.Dataset) -> pydicom.Dataset:
        """Transcode to Explicit VR Little Endian if dataset is compressed or not in an uncompressed TS."""
        try:
            ts = getattr(ds.file_meta, "TransferSyntaxUID", None)
            if not ts:
                return ds
            # Treat as uncompressed only for the known uncompressed transfer syntaxes
            uncompressed = {
                str(ExplicitVRLittleEndian),
                str(ImplicitVRLittleEndian),
                "1.2.840.10008.1.2.2",  # Explicit VR Big Endian (retired but uncompressed)
            }
            if str(ts) not in uncompressed:
                # Decompress pixel data in place (requires pylibjpeg[all] for JPEG/J2K)
                ds.decompress()
                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        except Exception as exc:  # pylint: disable=broad-except
            orthanc.LogWarning(f"Failed to decompress DICOM: {exc}")
        return ds

    def _build_uncompressed_contexts(self):
        """Requested contexts restricted to uncompressed syntaxes for compatibility."""
        syntaxes = [ExplicitVRLittleEndian, ImplicitVRLittleEndian]
        try:
            return [
                build_context(cx.abstract_syntax, syntaxes)
                for cx in StoragePresentationContexts
            ]
        except Exception:
            return StoragePresentationContexts

    def _requested_contexts_for_dataset(self, ds: pydicom.Dataset):
        """Prefer a single context matching the dataset SOP Class with uncompressed syntaxes."""
        syntaxes = [ExplicitVRLittleEndian, ImplicitVRLittleEndian]
        try:
            sop = getattr(ds, "SOPClassUID", None)
            if sop:
                return [build_context(sop, syntaxes)]
        except Exception:
            pass
        return self._build_uncompressed_contexts()

    def send_to_xnat(self, ds: pydicom.Dataset) -> bool:
        """Send the DICOM dataset to XNAT using C-STORE with retries and safe transfer syntax.
        Returns True on success, False otherwise."""
        # Ensure uncompressed syntax for broader compatibility
        ds = self._ensure_uncompressed(ds)

        attempts = 0
        max_attempts = 3
        backoff = 1.0
        while attempts < max_attempts:
            attempts += 1
            ae = None
            try:
                ae = AE(ae_title=self.ae_title)
                # Restrict to uncompressed explicit little endian
                ae.requested_contexts = self._requested_contexts_for_dataset(ds)
                # Reasonable timeouts for long runs
                ae.acse_timeout = 30
                ae.dimse_timeout = 60
                ae.network_timeout = 30

                assoc: Association = ae.associate(
                    self.xnat_ip, self.xnat_port, ae_title=self.xnat_ae_title
                )

                if assoc.is_established:
                    status = assoc.send_c_store(ds)
                    assoc.release()
                    # status is a pydicom Dataset; success is 0x0000
                    try:
                        code = getattr(status, "Status", None)
                    except Exception:
                        code = None
                    if code == 0x0000:
                        orthanc.LogInfo("Successfully sent DICOM file to XNAT")
                        return True
                    else:
                        orthanc.LogError(f"C-STORE failed with status: {code}")
                else:
                    orthanc.LogError("Failed to associate with XNAT")
            except Exception as exc:  # pylint: disable=broad-except
                orthanc.LogError(f"Exception during C-STORE: {exc}")
            finally:
                try:
                    if ae is not None:
                        ae.shutdown()
                except Exception:
                    pass

            # Retry with backoff
            time.sleep(backoff)
            backoff = min(backoff * 2, 10)
        return False

    @staticmethod
    def _extract_patient_ids(ds: pydicom.Dataset) -> List[str]:
        """Extract possible alternate patient identifiers from standard fields.
        Returns a list of candidate IDs (may be empty)."""

        def _to_list(v) -> List[str]:
            if v is None:
                return []
            # Treat any non-string sequence (e.g., pydicom MultiValue) as a list of values
            try:
                from collections.abc import Sequence as _Seq

                if isinstance(v, _Seq) and not isinstance(v, (str, bytes)):
                    return [str(x) for x in v if x is not None and str(x) != ""]
            except Exception:
                pass
            s = str(v).strip()
            # Try to parse Python-like list strings: "['A','B']"
            if s.startswith("[") and s.endswith("]"):
                try:
                    lit = ast.literal_eval(s)
                    if isinstance(lit, (list, tuple)):
                        return [str(x) for x in lit if x is not None and str(x) != ""]
                except Exception:
                    pass
            # Split by DICOM multi-valued delimiter (backslash)
            if "\\" in s:
                return [p.strip() for p in s.split("\\") if p.strip()]
            # As a last resort, split by comma if present
            if "," in s:
                return [p.strip().strip("'\"") for p in s.split(",") if p.strip()]
            return [s] if s else []

        candidates: List[str] = []
        # Primary ID(s)
        candidates.extend(_to_list(ds.get("PatientID")))
        # OtherPatientIDs may be multi-valued
        candidates.extend(_to_list(ds.get("OtherPatientIDs")))
        # OtherPatientIDsSequence items may contain IDs
        seq = ds.get("OtherPatientIDsSequence")
        if seq:
            try:
                for item in seq:
                    candidates.extend(_to_list(item.get("PatientID")))
                    candidates.extend(_to_list(item.get("ID")))
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

    def _queue_pending_instance(
        self, ids: List[str], instance_id: Optional[str]
    ) -> None:
        """Queue this instance to be retried later when a mapping is discovered for any of the given ids."""
        if not instance_id:
            orthanc.LogWarning(
                "Instance has no string ID; cannot queue for retry without mapping"
            )
            return
        with self._pending_lock:
            # Per-instance retry cap
            count = self._pending_retries.get(instance_id, 0)
            if count >= self._pending_retries_max:
                orthanc.LogWarning(
                    f"Max pending retries reached for {instance_id}; dropping from queue"
                )
                return
            self._pending_retries[instance_id] = count + 1

            if len(self._pending_by_id) >= self._pending_max_ids:
                orthanc.LogWarning(
                    "Pending mapping queue is full; dropping new pending item"
                )
                return
            for pid in ids:
                try:
                    pid = str(pid)
                except Exception:  # pylint: disable=broad-except
                    continue
                s = self._pending_by_id.get(pid)
                if s is None:
                    s = set()
                    self._pending_by_id[pid] = s
                s.add(instance_id)
        orthanc.LogInfo(
            f"Queued instance {instance_id} for retry; awaiting mapping for IDs: {ids}"
        )

    def _submit_instance(self, instance_id: str) -> None:
        """Submit an instance to the executor with queue bounding."""
        self._submit_sema.acquire()

        def _run():
            try:
                self.process_instance(instance_id)
            finally:
                try:
                    self._submit_sema.release()
                except Exception:
                    pass

        self.executor.submit(_run)

    def _flush_pending_for_ids(self, ids: List[str]) -> None:
        """Re-schedule any pending instances that share any of these ids now that a mapping is known."""
        to_retry: set[str] = set()
        with self._pending_lock:
            for pid in ids:
                pid_str = str(pid)
                instances = self._pending_by_id.pop(pid_str, None)
                if instances:
                    to_retry.update(instances)
        if to_retry:
            orthanc.LogInfo(
                f"Discovered mapping; retrying {len(to_retry)} pending instance(s)"
            )
            for inst_id in to_retry:
                try:
                    self._submit_instance(inst_id)
                except Exception as exc:  # pylint: disable=broad-except
                    orthanc.LogError(
                        f"Failed to resubmit pending instance {inst_id}: {exc}"
                    )

    @staticmethod
    def _sanitize_dataset(ds: pydicom.Dataset) -> pydicom.Dataset:
        """Coerce common invalid VR values to valid forms to reduce warnings and improve interoperability."""

        def _sanitize_is(val):
            def _one(x):
                # Convert floats/strings like '120000.000000' to '120000', strip non-digits
                if isinstance(x, (int, float)):
                    return str(int(x))
                s = str(x)
                m = re.search(r"[-+]?\d+", s)
                if m:
                    out = m.group(0)
                else:
                    out = "0"
                return out[:12]

            if isinstance(val, (list, tuple)):
                return [_one(v) for v in val]
            return _one(val)

        def _limit_len(val, n, upper=False, allowed=None):
            def _one(x):
                s = str(x)
                if upper:
                    s = s.upper()
                if allowed is not None:
                    s = "".join(ch for ch in s if ch in allowed)
                return s[:n]

            if isinstance(val, (list, tuple)):
                return [_one(v) for v in val]
            return _one(val)

        # Allowed chars for CS
        cs_allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _")

        try:
            for elem in ds.iterall():
                vr = getattr(elem, "VR", None)
                if not vr:
                    continue
                if vr == "IS":
                    try:
                        elem.value = _sanitize_is(elem.value)
                    except Exception:
                        pass
                elif vr == "CS":
                    try:
                        elem.value = _limit_len(
                            elem.value, 16, upper=True, allowed=cs_allowed
                        )
                    except Exception:
                        pass
                elif vr == "SH":
                    try:
                        elem.value = _limit_len(elem.value, 16)
                    except Exception:
                        pass
                elif vr == "LO":
                    try:
                        elem.value = _limit_len(elem.value, 64)
                    except Exception:
                        pass
        except Exception:
            # Best effort sanitization; ignore if iterating fails
            pass
        return ds

    def process_instance(self, instance: Union[orthanc.DicomInstance, str]) -> None:
        """Process a new DICOM instance by re-identifying and sending it to XNAT."""
        try:
            instance_id_str: Optional[str] = (
                instance if isinstance(instance, str) else None
            )
            if isinstance(instance, str):
                dcm_bytes = orthanc.GetDicomForInstance(instance)
            else:
                dcm_bytes = instance.GetInstanceData()
            if not dcm_bytes:
                orthanc.LogError(f"Could not get DICOM bytes for instance {instance}")
                return

            try:
                # Suppress noisy warnings emitted during parsing; we sanitize right after
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
                ds.filename = None
            except pydicom.errors.InvalidDicomError:
                orthanc.LogError(f"Invalid DICOM instance: {instance}")
                return

            # Sanitize common VR issues before further processing
            ds = self._sanitize_dataset(ds)

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
            if not new_patient_id or key == new_patient_id:
                # No mapping yet; queue for retry and return without sending
                self._queue_pending_instance(all_ids, instance_id_str)
                orthanc.LogWarning(
                    f"No mapping found yet for IDs {all_ids}; queued instance for retry"
                )
                return

            # Mapping discovered; seed cache for all observed IDs and retry any pending
            try:
                formatted_ids = [formatting.format(i) for i in all_ids if i]
                self.excel_client.set_mapping(sheet, formatted_ids, new_patient_id)
            except Exception:  # pylint: disable=broad-except
                pass
            self._flush_pending_for_ids(all_ids)

            deidentified_ds = self.dicom_processor.deidentify_dicom(ds, new_patient_id)
            orthanc.LogInfo(
                f"Re-identified DICOM with new patient ID: {new_patient_id}"
            )
            ok = self.send_to_xnat(deidentified_ds)
            if ok and instance_id_str:
                # Clear retry counter on success
                with self._pending_lock:
                    self._pending_retries.pop(instance_id_str, None)
        except Exception as exc:  # pylint: disable=broad-except
            orthanc.LogError(
                f"Unhandled exception while processing instance {instance}: {exc}"
            )

    def on_stored_instance(self, instance_id: str, *args, **kwargs) -> None:
        """Callback function triggered when a new instance is stored in Orthanc."""
        try:
            orthanc.LogInfo(f"Queueing new instance: {instance_id}")
            self._submit_instance(instance_id)
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
                self._submit_instance(inst)
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
                        self._submit_instance(inst)
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
