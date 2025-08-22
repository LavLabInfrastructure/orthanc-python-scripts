"""Defines OrthancCallbackHandler which handles callbacks for the Orthanc plugin."""

import io
import json
import threading
import time
import warnings
import re
import ast
from typing import Union, List, Optional, Any
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

        # Reusable, bounded worker pools
        max_workers = int(orthanc_config.get("max_workers", 8) or 8)
        cpu_workers = int(orthanc_config.get("cpu_max_workers", max_workers) or max_workers)
        io_workers = int(orthanc_config.get("io_max_workers", max_workers * 2) or (max_workers * 2))
        self.executor = ThreadPoolExecutor(max_workers=cpu_workers)
        self._io_executor = ThreadPoolExecutor(max_workers=io_workers)
        # Dedicated send executor (I/O bound)
        send_workers = int(orthanc_config.get("send_max_workers", max_workers) or max_workers)
        self.send_executor = ThreadPoolExecutor(max_workers=send_workers)
        # Bound the number of queued tasks to protect memory over multi-day runs
        max_queue = int(orthanc_config.get("max_queue", 1000) or 1000)
        self._submit_sema = threading.Semaphore(max_queue)
        io_max_queue = int(orthanc_config.get("io_max_queue", max_queue) or max_queue)
        self._io_submit_sema = threading.Semaphore(io_max_queue)
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
        # Prefer keeping images compressed end-to-end; include compressed syntaxes in contexts
        self._prefer_compressed = str(orthanc_config.get("prefer_compressed", "true")).lower() not in ("false", "0", "no")
        # Memoize preferred contexts (compressed + uncompressed) and configure association pool
        self._contexts_all_preferred = self._build_preferred_contexts()
        # PDU/assoc config
        self._assoc_pool_size = int(
            orthanc_config.get("assoc_pool_size", max_workers) or max_workers
        )
        # Use a larger default PDU; negotiated down by peer as needed
        self._max_pdu_size = int(orthanc_config.get("max_pdu_size", 4194304) or 4194304)
        self._acse_timeout = int(orthanc_config.get("acse_timeout", 30) or 30)
        self._dimse_timeout = int(orthanc_config.get("dimse_timeout", 60) or 60)
        self._network_timeout = int(orthanc_config.get("network_timeout", 30) or 30)
        self._assoc_pool = self._AssociationPool(
            self.xnat_ip,
            self.xnat_port,
            self.ae_title,
            self.xnat_ae_title,
            self._contexts_all_preferred,
            self._assoc_pool_size,
            self._max_pdu_size,
            self._acse_timeout,
            self._dimse_timeout,
            self._network_timeout,
        )
        # Per-thread association reuse (faster and simpler than pooling)
        self._per_thread_assoc = str(orthanc_config.get("per_thread_association", "true")).lower() not in ("false", "0", "no")
        self._tls = threading.local()

    class _AssociationPool:
        """Thread-safe pool of persistent pynetdicom associations."""

        def __init__(
            self,
            ip: str,
            port: int,
            local_ae_title: str,
            peer_ae_title: str,
            requested_contexts,
            pool_size: int,
            max_pdu_size: int,
            acse_timeout: int,
            dimse_timeout: int,
            network_timeout: int,
        ):
            self._ip = ip
            self._port = port
            self._local = local_ae_title
            self._peer = peer_ae_title
            self._contexts = requested_contexts
            self._pool_size = max(1, pool_size)
            self._max_pdu = max_pdu_size
            self._acse = acse_timeout
            self._dimse = dimse_timeout
            self._net = network_timeout
            self._lock = threading.Lock()
            self._available: list[Association] = []
            self._total = 0

        def _create(self) -> Optional[Association]:
            try:
                ae = AE(ae_title=self._local)
                ae.requested_contexts = self._contexts
                ae.maximum_pdu_size = self._max_pdu
                ae.acse_timeout = self._acse
                ae.dimse_timeout = self._dimse
                ae.network_timeout = self._net
                assoc: Association = ae.associate(
                    self._ip, self._port, ae_title=self._peer
                )
                if assoc.is_established:
                    # attach AE so we can shutdown when discarding
                    setattr(assoc, "_lavlab_ae", ae)
                    return assoc
                else:
                    try:
                        ae.shutdown()
                    except Exception:
                        pass
                    return None
            except Exception as exc:
                orthanc.LogError(f"Failed to create association: {exc}")
                return None

        def acquire(self, timeout: float = 0.0) -> Optional[Association]:
            """Get an established association or create one; blocks if pool is full and none available when timeout<=0."""
            end = None if timeout <= 0 else (time.time() + timeout)
            while True:
                with self._lock:
                    if self._available:
                        assoc = self._available.pop()
                        if assoc.is_established:
                            return assoc
                        else:
                            self._discard_internal(assoc)
                    elif self._total < self._pool_size:
                        assoc = self._create()
                        if assoc is not None:
                            self._total += 1
                            return assoc
                # outside lock
                if end is not None and time.time() >= end:
                    return None
                time.sleep(0.01)

        def release(self, assoc: Association) -> None:
            with self._lock:
                if assoc and assoc.is_established:
                    self._available.append(assoc)
                else:
                    self._discard_internal(assoc)

        def discard(self, assoc: Association) -> None:
            with self._lock:
                self._discard_internal(assoc)

        def _discard_internal(self, assoc: Optional[Association]) -> None:
            try:
                if assoc is not None:
                    try:
                        assoc.abort()
                    except Exception:
                        pass
                    try:
                        ae = getattr(assoc, "_lavlab_ae", None)
                        if ae is not None:
                            ae.shutdown()
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                self._total = max(0, self._total - 1)

    @staticmethod
    def _ensure_uncompressed(ds: pydicom.Dataset) -> pydicom.Dataset:
        """If prefer_compressed is enabled, do nothing; otherwise, transcode to Explicit VR Little Endian when needed."""
        # No-op here; decompression is managed by preference in requested contexts
        return ds

    def _build_preferred_contexts(self):
        """Requested contexts including common compressed syntaxes plus uncompressed fallbacks."""
        compressed = [
            "1.2.840.10008.1.2.4.70",  # JPEG Lossless, Non-Hierarchical, P14 SV1
            "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless
            "1.2.840.10008.1.2.4.91",  # JPEG 2000
            "1.2.840.10008.1.2.4.50",  # JPEG Baseline
            "1.2.840.10008.1.2.4.51",  # JPEG Extended
            "1.2.840.10008.1.2.5",     # RLE Lossless
        ]
        syntaxes = compressed + [str(ExplicitVRLittleEndian), str(ImplicitVRLittleEndian)]
        try:
            return [
                build_context(cx.abstract_syntax, syntaxes)
                for cx in StoragePresentationContexts
            ]
        except Exception:
            return StoragePresentationContexts

    def _requested_contexts_for_dataset(self, ds: pydicom.Dataset):
        """Prefer a single context matching the dataset SOP Class with preferred syntaxes including source TS."""
        compressed = [
            "1.2.840.10008.1.2.4.70",
            "1.2.840.10008.1.2.4.90",
            "1.2.840.10008.1.2.4.91",
            "1.2.840.10008.1.2.4.50",
            "1.2.840.10008.1.2.4.51",
            "1.2.840.10008.1.2.5",
        ]
        try:
            sop = getattr(ds, "SOPClassUID", None)
            ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
            if sop:
                ordered = []
                if ts:
                    ordered.append(str(ts))
                ordered.extend(compressed)
                ordered.extend([str(ExplicitVRLittleEndian), str(ImplicitVRLittleEndian)])
                # Deduplicate preserving order
                seen = set()
                prefer = [x for x in ordered if not (x in seen or seen.add(x))]
                return [build_context(sop, prefer)]
        except Exception:
            pass
        return self._contexts_all_preferred

    def _send_with_ephemeral_assoc(self, ds: pydicom.Dataset) -> bool:
        """Send using a one-off association with SOP-specific contexts (no pool wait)."""
        try:
            ae = AE(ae_title=self.ae_title)
            ae.requested_contexts = self._requested_contexts_for_dataset(ds)
            ae.maximum_pdu_size = self._max_pdu_size
            ae.acse_timeout = self._acse_timeout
            ae.dimse_timeout = self._dimse_timeout
            ae.network_timeout = self._network_timeout
            assoc: Association = ae.associate(self.xnat_ip, self.xnat_port, ae_title=self.xnat_ae_title)
            if not assoc.is_established:
                try:
                    ae.shutdown()
                except Exception:
                    pass
                return False
            try:
                status = assoc.send_c_store(ds)
                code = getattr(status, 'Status', None)
                assoc.release()
                return code == 0x0000
            finally:
                try:
                    ae.shutdown()
                except Exception:
                    pass
        except Exception:
            return False

    def _get_thread_assoc(self) -> Optional[Association]:
        """Get or create a persistent association for the current thread."""
        try:
            assoc = getattr(self._tls, "assoc", None)
            if assoc is not None and assoc.is_established:
                return assoc
            ae = AE(ae_title=self.ae_title)
            ae.requested_contexts = self._contexts_all_preferred
            ae.maximum_pdu_size = self._max_pdu_size
            ae.acse_timeout = self._acse_timeout
            ae.dimse_timeout = self._dimse_timeout
            ae.network_timeout = self._network_timeout
            assoc: Association = ae.associate(self.xnat_ip, self.xnat_port, ae_title=self.xnat_ae_title)
            if assoc.is_established:
                self._tls.assoc = assoc
                self._tls.ae = ae
                return assoc
            try:
                ae.shutdown()
            except Exception:
                pass
            return None
        except Exception:
            return None

    def _drop_thread_assoc(self) -> None:
        """Drop the current thread's association and shutdown AE."""
        try:
            assoc = getattr(self._tls, "assoc", None)
            if assoc is not None:
                try:
                    assoc.abort()
                except Exception:
                    pass
            ae = getattr(self._tls, "ae", None)
            if ae is not None:
                try:
                    ae.shutdown()
                except Exception:
                    pass
            self._tls.assoc = None
            self._tls.ae = None
        except Exception:
            pass

    def _send_dataset(self, ds: pydicom.Dataset, instance_id_str: Optional[str]) -> None:
        """Send dataset using the I/O executor and clear retry counters on success."""
        ok = self.send_to_xnat(ds)
        if ok and instance_id_str:
            with self._pending_lock:
                self._pending_retries.pop(instance_id_str, None)

    def send_to_xnat(self, ds: pydicom.Dataset) -> bool:
        """Send the DICOM dataset to XNAT using C-STORE; prefer per-thread association reuse, with fallbacks.
        Tries original transfer syntax first; if negotiation fails, decompress and retry uncompressed.
        """
        attempts = 0
        max_attempts = 3
        backoff = 0.5
        while attempts < max_attempts:
            attempts += 1
            # Fast path: thread-local association reuse
            if self._per_thread_assoc:
                assoc = self._get_thread_assoc()
                if assoc is not None:
                    try:
                        status = assoc.send_c_store(ds)
                        code = getattr(status, "Status", None)
                        if code == 0x0000:
                            orthanc.LogInfo("Successfully sent DICOM file to XNAT")
                            return True
                        # If failed due to presentation context for current TS, decompress and retry once
                        self._drop_thread_assoc()
                    except Exception as exc:  # pylint: disable=broad-except
                        msg = str(exc)
                        # Try fallback: decompress and retry fresh assoc
                        try:
                            new_ds = self._ensure_uncompressed(ds)
                            a = self._get_thread_assoc()
                            if a is not None:
                                st = a.send_c_store(new_ds)
                                if getattr(st, 'Status', None) == 0x0000:
                                    return True
                        except Exception:
                            pass
                    # Retry loop will continue
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5)
                    continue
            # Fallback to pool/ephemeral path
            assoc = self._assoc_pool.acquire(timeout=0.05)
            if assoc is None:
                # Ephemeral try with preferred contexts
                if self._send_with_ephemeral_assoc(ds):
                    orthanc.LogInfo("Successfully sent via ephemeral association")
                    return True
                # Decompress and try ephemeral uncompressed
                try:
                    new_ds = self._ensure_uncompressed(ds)
                    if self._send_with_ephemeral_assoc(new_ds):
                        orthanc.LogInfo("Successfully sent via uncompressed ephemeral association")
                        return True
                except Exception:
                    pass
                time.sleep(backoff)
                backoff = min(backoff * 2, 5)
                continue
            try:
                status = assoc.send_c_store(ds)
                code = getattr(status, "Status", None)
                if code == 0x0000:
                    self._assoc_pool.release(assoc)
                    orthanc.LogInfo("Successfully sent DICOM file to XNAT")
                    return True
                else:
                    # Try decompress on same association (it likely accepted uncompressed contexts too)
                    try:
                        new_ds = self._ensure_uncompressed(ds)
                        st2 = assoc.send_c_store(new_ds)
                        if getattr(st2, 'Status', None) == 0x0000:
                            self._assoc_pool.release(assoc)
                            return True
                    except Exception:
                        pass
                    orthanc.LogError(f"C-STORE failed with status: {code}")
                    self._assoc_pool.discard(assoc)
            except Exception as exc:  # pylint: disable=broad-except
                # Decompress and retry once if possible
                try:
                    new_ds = self._ensure_uncompressed(ds)
                    st3 = assoc.send_c_store(new_ds)
                    if getattr(st3, 'Status', None) == 0x0000:
                        self._assoc_pool.release(assoc)
                        return True
                except Exception:
                    pass
                orthanc.LogError(f"Exception during C-STORE: {exc}")
                self._assoc_pool.discard(assoc)
            time.sleep(backoff)
            backoff = min(backoff * 2, 5)
        return False

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
                return [_sanitize_is(v) for v in val]
            return _sanitize_is(val)

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
                elif vr == "UI":
                    try:
                        val = str(elem.value)
                        new = "".join(ch for ch in val if (ch.isdigit() or ch == "."))[:64]
                        new = new.rstrip(".")
                        if new:
                            elem.value = new
                    except Exception:
                        pass
        except Exception:
            pass
        return ds

    def _validate_before_send(self, ds: pydicom.Dataset) -> bool:
        """Validate essential tags before sending to XNAT; fix trivially fixable issues.
        Returns True if OK to send, False to skip.
        """

        def _is_valid_ui(s: Any) -> bool:
            if not s:
                return False
            ss = str(s)
            if len(ss) > 64:
                return False
            if ss.endswith("."):
                return False
            return all(ch.isdigit() or ch == "." for ch in ss)

        required = [
            "SOPClassUID",
            "SOPInstanceUID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
        ]
        ok = True
        for tag in required:
            val = ds.get(tag)
            if not _is_valid_ui(val):
                # Try to repair deterministically from whatever we have
                base = f"{tag}:{ds.get('PatientID','')}:${ds.get('StudyDate','')}:{ds.get('Modality','')}:{ds.get('AccessionNumber','')}"
                try:
                    new_uid = DicomProcessor.hash_dicom_uid(base)
                    setattr(ds, tag, new_uid)
                    orthanc.LogWarning(f"Repaired invalid/missing {tag} -> {new_uid}")
                except Exception:
                    ok = False
        # Minimal image consistency: if PixelData present, ensure Rows/Columns present
        if hasattr(ds, "PixelData"):
            if ds.get("Rows") is None or ds.get("Columns") is None:
                orthanc.LogWarning(
                    "Dataset has PixelData but missing Rows/Columns; skipping send"
                )
                ok = False
        return ok

    def _extract_patient_ids(self, ds: pydicom.Dataset) -> List[str]:
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

            # Header-only fast read to avoid touching pixel data if we won't send
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    ds_hdr = pydicom.dcmread(io.BytesIO(dcm_bytes), stop_before_pixels=True, defer_size="1 KB")
                ds_hdr.filename = None
            except pydicom.errors.InvalidDicomError:
                orthanc.LogError(f"Invalid DICOM instance: {instance}")
                return

            # Use header dataset for matching and mapping
            sheet = self.dicom_processor.match_dicom_to_sheet(ds_hdr)
            formatting = sheet.get("format", "{}")
            all_ids = self._extract_patient_ids(ds_hdr)
            if not all_ids:
                orthanc.LogWarning(
                    "No patient identifiers found in DICOM; using PatientID as-is"
                )
                all_ids = [str(ds_hdr.get("PatientID", ""))]
            key = (
                formatting.format(all_ids[0])
                if all_ids[0]
                else formatting.format(ds_hdr.get("PatientID", ""))
            )
            alt_keys = [formatting.format(i) for i in all_ids[1:]]

            new_patient_id = self.excel_client.get_patient_id(key, sheet, alt_keys)
            if not new_patient_id or key == new_patient_id:
                self._queue_pending_instance(all_ids, instance_id_str)
                orthanc.LogWarning(
                    f"No mapping found yet for IDs {all_ids}; queued instance for retry"
                )
                return

            # Full read only when we plan to send; still defer large attributes where possible
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                ds = pydicom.dcmread(io.BytesIO(dcm_bytes), defer_size="1 KB")
            ds.filename = None

            # Deidentify and sanitize, then schedule send on I/O executor
            deidentified_ds = self.dicom_processor.deidentify_dicom(ds, new_patient_id)
            deidentified_ds = self._sanitize_dataset(deidentified_ds)
            self.send_executor.submit(self._send_dataset, deidentified_ds, instance_id_str)
        except Exception as exc:  # pylint: disable=broad-except
            orthanc.LogError(
                f"Unhandled exception while processing instance {instance}: {exc}"
            )

    def _submit_send(self, ds: pydicom.Dataset, instance_id_str: Optional[str]) -> None:
        """Submit a send task to the IO executor with queue bounding."""
        self._io_submit_sema.acquire()
        def _run_send():
            try:
                ok = self.send_to_xnat(ds)
                if ok and instance_id_str:
                    with self._pending_lock:
                        self._pending_retries.pop(instance_id_str, None)
            except Exception:
                pass
            finally:
                try:
                    self._io_submit_sema.release()
                except Exception:
                    pass
        self._io_executor.submit(_run_send)

    def _submit_instance(self, instance_id: str) -> None:
        """Submit a processing task to the CPU executor with queue bounding."""
        self._submit_sema.acquire()
        def _run_process():
            try:
                self.process_instance(instance_id)
            finally:
                try:
                    self._submit_sema.release()
                except Exception:
                    pass
        self.executor.submit(_run_process)

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
