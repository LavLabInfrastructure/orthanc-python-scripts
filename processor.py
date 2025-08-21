"""Defines DicomProcessor class for de-identifying DICOM datasets."""

import re
import base64
import hashlib
from typing import Any, Optional
from datetime import datetime

import pydicom
import pydicom.sequence
from deid.config import DeidRecipe
from deid.dicom.parser import DicomParser

# orthanc doesn't have a pypi package, so pylance complains about the import, works fine at runtime
# if you want autocomplete, add orthanc.pyi to the directory, downloaded from the link below
# https://orthanc.uclouvain.be/downloads/cross-platform/orthanc-python/index.html
import orthanc  # type: ignore # pylint: disable=import-error

from config import Config


# pylint: disable=unused-argument
class DicomProcessor:
    """Handles de-identification of DICOM datasets."""

    def __init__(self, config: Config):
        """Initialize the DicomProcessor with the given configuration."""
        orthanc_config = config.get_orthanc_config()
        deid_config = orthanc_config["deid"]
        self.recipe = DeidRecipe(
            deid_config["recipe_paths"], deid_config["use_base_recipe"]
        )
        self.sheets = config.get_sheets()
        # Precompile identifier regex for speed
        for sheet in self.sheets:
            compiled_list = []
            for identifier in sheet.get("identifiers", []):
                pattern = identifier.get("regex")
                tag = identifier.get("tag")
                if not tag or not pattern:
                    continue
                try:
                    cre = re.compile(pattern)
                except re.error:
                    cre = None
                identifier["_regex_compiled"] = cre
                compiled_list.append((tag, cre, pattern))
            sheet["_compiled_identifiers"] = compiled_list
        # Build a stable ordered list of tags used across all sheets for fingerprinting
        tag_order = []
        seen_tags = set()
        for sheet in self.sheets:
            for tag, _, _ in sheet.get("_compiled_identifiers", []):
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    tag_order.append(tag)
        self._match_tags: list[str] = tag_order
        # Simple LRU-like cache for match results
        self._match_cache: dict[tuple, dict[str, Any]] = {}
        self._match_cache_max = 8192

    def deidentify_dicom(
        self, ds: pydicom.Dataset, patient_id: str, sheet: Optional[dict[str, Any]] = None
    ) -> pydicom.Dataset:
        """De-identify the DICOM dataset and return the de-identified dataset.
        Optionally pass a pre-matched `sheet` to avoid recomputation inside helpers.
        """
        if sheet is None:
            # Best effort: compute and cache on dataset for helper functions
            try:
                sheet = self.match_dicom_to_sheet(ds)
            except Exception:
                sheet = None
        # Cache selected sheet on the dataset for downstream helpers
        try:
            if sheet is not None:
                setattr(ds, "_lavlab_sheet", sheet)
        except Exception:
            pass

        parser = DicomParser(ds, self.recipe)

        parser.define("excel_id", patient_id)
        parser.define("format_xnat_route", self.format_xnat_route)
        parser.define("format_xnat_route_antiphi", self.format_xnat_route_antiphi)
        parser.define("hash_uid_func", self.deid_hash_uid_func)
        parser.define("remove_day", self.remove_day)
        parser.define("hash_func", self.deid_hash_func)
        parser.define("gather_diffusion_tags", self.gather_diffusion_tags)
        parser.define("round_func", self.deid_round_func)

        parser.parse(strip_sequences=False, remove_private=False)
        parser.remove_private()
        return parser.dicom

    def format_xnat_route_antiphi(self, item, value, field, dicom):
        """
        Formats the XNAT route and assigns to the given field.
        This variant checks for KO and SR modalities and ConversionType
        to determine if the session is PHI.
        """

        xnat_project = self.get_project(dicom)
        # logic to remove icky stuff
        modality = dicom.get("Modality")
        if modality == "KO" or modality == "SR":
            return "Project: NA Subject: HIDDEN Session: PROBABLE_PHI"
        elif dicom.get("ConversionType") is not None:
            return "Project: NA Subject: HIDDEN Session: PROBABLE_PHI"
        sesh = (
            f"{dicom.get('PatientID')}_{dicom.get('StudyDate')}_{dicom.get('Modality')}"
        )

        return (
            f"Project: {xnat_project} Subject: {dicom.get('PatientID')} Session: {sesh}"
        )

    def format_xnat_route(self, item, value, field, dicom):
        """Formats the XNAT route and assigns to the given field."""
        xnat_project = self.get_project(dicom)
        sesh = (
            f"{dicom.get('PatientID')}_{dicom.get('StudyDate')}_{dicom.get('Modality')}"
        )
        return (
            f"Project: {xnat_project} Subject: {dicom.get('PatientID')} Session: {sesh}"
        )

    def get_project(self, ds: pydicom.Dataset):
        """Use matching sheet to determine XNAT project, with per-dataset caching."""
        try:
            sheet = getattr(ds, "_lavlab_sheet", None)
            if sheet is None:
                sheet = self.match_dicom_to_sheet(ds)
                setattr(ds, "_lavlab_sheet", sheet)
            return sheet["xnat_project"]
        except Exception:
            sheet = self.match_dicom_to_sheet(ds)
            return sheet["xnat_project"]

    def match_dicom_to_sheet(self, ds: pydicom.Dataset) -> dict[str, Any]:
        """Find the sheet for the given DICOM dataset based on regex matching with caching."""
        # Build fingerprint key from relevant tags
        fp_vals = []
        for tag in self._match_tags:
            v = ds.get(tag)
            if v is None:
                fp_vals.append(None)
            else:
                if not isinstance(v, str):
                    try:
                        v = v.value
                    except Exception:  # pylint: disable=broad-except
                        v = str(v)
                fp_vals.append(str(v))
        fp = tuple(fp_vals)
        hit = self._match_cache.get(fp)
        if hit is not None:
            return hit

        # Fall back to regex scanning
        for sheet in self.sheets:
            for tag, cre, pattern in sheet.get("_compiled_identifiers", []):
                tag_val = ds.get(tag)
                if tag_val is None:
                    continue
                if not isinstance(tag_val, str):
                    try:
                        tag_val = tag_val.value
                    except Exception:  # pylint: disable=broad-except
                        tag_val = str(tag_val)
                s = str(tag_val)
                try:
                    if cre is not None:
                        if cre.match(s):
                            # Cache and return
                            if len(self._match_cache) >= self._match_cache_max:
                                self._match_cache.clear()
                            self._match_cache[fp] = sheet
                            return sheet
                    else:
                        if re.match(pattern, s):
                            if len(self._match_cache) >= self._match_cache_max:
                                self._match_cache.clear()
                            self._match_cache[fp] = sheet
                            return sheet
                except re.error:
                    continue
        orthanc.LogError("No matching sheet found for the given DICOM dataset")
        # Default
        default_sheet = self.sheets[0]
        if len(self._match_cache) >= self._match_cache_max:
            self._match_cache.clear()
        self._match_cache[fp] = default_sheet
        return default_sheet

    def deid_hash_uid_func(self, item, value, field, dicom) -> str:
        """Performs self.hash to field.element.value"""
        val = field.element.value
        return DicomProcessor.hash_dicom_uid(str(val))

    def deid_hash_func(self, item, value, field, dicom) -> str:
        """Performs self.hash to field.element.value"""
        val = field.element.value
        return DicomProcessor.hash(str(val))[:15]

    def gather_diffusion_tags(self, item, value, field, dicom) -> pydicom.Dataset:
        """Gathers relevant diffusion info and formats into an MRDiffusionSequence."""
        desc_val = dicom.get('SeriesDescription', '')
        desc = str(desc_val).lower()
        # Require either 'diffusion' OR 'dwi', and exclude ADC series
        if ((('diffusion' not in desc) and ('dwi' not in desc)) or
            'adc' in desc or 'apparent diffusion coefficient' in desc):
            return None

        manufacturer = str(dicom.get('Manufacturer', '')).upper() if 'Manufacturer' in dicom else ''
        if not manufacturer:
            # Missing manufacturer; skip gracefully
            return None

        handler = self.MANUFACTURER_HANDLERS.get(manufacturer)
        if not handler:
            # Unsupported manufacturer; skip gracefully
            return None

        # Call the handler function to get DiffusionSequence
        return handler(dicom)

    def deid_round_func(self, item, value, field, dicom) -> str:
        """Performs self.hash to field.element.value"""
        val = str(field.element.value).lower().strip("y")
        return f"{self.round_to_nearest_five(int(val)):03}Y"

    @staticmethod
    def remove_day(item, value, field, dicom) -> str:
        """Removes the day from a DT field in the deid framework"""
        dt = datetime.strptime(field.element.value, "%Y%m%d")
        return dt.strftime("%Y%m01")

    @staticmethod
    def hash_dicom_uid(uid: str, namespace: str = "1.2.840.113619") -> str:
        """
        Hashes a DICOM UID using SHA-256, truncates the hash, encodes it in Base32,
        and converts it to a valid numeric UID.

        Args:
            uid (str): The DICOM UID to hash.
            namespace (str): The namespace prefix for the hashed UID.

        Returns:
            str: The hashed and formatted UID.
        """
        sha256 = hashlib.sha256()
        sha256.update(uid.encode("utf-8"))
        hash_bytes = sha256.digest()

        truncated_hash_bytes = hash_bytes[:24]

        base32_hash = base64.b32encode(truncated_hash_bytes).decode("utf-8").rstrip("=")

        numeric_uid = "".join(
            str(ord(c) - ord("A") + 10) if "A" <= c <= "Z" else c for c in base32_hash
        )  # pylint: disable=line-too-long

        max_numeric_length = 64 - len(namespace) - 1
        numeric_uid = numeric_uid[:max_numeric_length]

        return f"{namespace}.{numeric_uid}"

    @staticmethod
    def hash(uid: str) -> str:
        """
        Hashes a DICOM UID using SHA-256 and returns the hexadecimal representation of the hash.

        Args:
            uid (str): The DICOM UID to hash.

        Returns:
            str: The SHA-256 hash of the UID.
        """
        sha256 = hashlib.sha256()
        sha256.update(uid.encode("utf-8"))
        return sha256.hexdigest()

    @staticmethod
    def round_to_nearest_five(n):
        """Rounds a number to the nearest multiple of 5."""
        return 5 * round(n / 5)

    @staticmethod
    def handle_siemens(ds: pydicom.Dataset) -> pydicom.Sequence:
        """Handles Siemens specific diffusion tags."""
        tags = pydicom.Dataset()
        # bval
        if (0x0019, 0x100C) in ds:
            value = float(ds[(0x0019, 0x100C)].value)
            tags.add_new((0x0018, 0x9087), "FD", value)

        # directionality
        if (0x0019, 0x100D) in ds:
            value = str(ds[(0x0019, 0x100D)].value)
            tags.add_new((0x0018, 0x9075), "CS", value.upper())

        # gradient direction and sequence (gradient orientation)
        if (0x0019, 0x100E) in ds:
            value = ds[(0x0019, 0x100E)].value
            _tag = pydicom.Dataset()
            _tag.add_new((0x0018, 0x9089), "FD", value)
            tags.add_new((0x0018, 0x9076), "SQ", pydicom.Sequence([_tag]))

        # bmatrix and sequence
        if (0x0019, 0x1027) in ds:
            value = ds[(0x0019, 0x1027)].value
            # should really be 6, hopefully this never hits
            if len(value) != 6:
                return None
            # decode bmatrix into their appropriate tags
            _tags = pydicom.Dataset()
            tag = 0x9602
            for val in value:
                _tags.add_new((0x0018, tag), "FD", val)
                tag += 1
            tags.add_new((0x0018, 0x9601), "SQ", pydicom.Sequence([_tags]))

        return pydicom.Sequence([tags])

    @staticmethod
    def handle_ge(ds: pydicom.Dataset) -> None:
        """Handles GE specific diffusion tags."""
        tags = pydicom.Dataset()
        # bval
        if (0x0043, 0x1039) in ds:
            value = int(ds[(0x0043, 0x1039)].value[0])
            if value > 10000:
                value = value % 100000  # god GE is so annoying
            tags.add_new((0x0018, 0x9087), "FD", value)

        ## GE DTI SUCKS EVEN MORE! waiting until we need it before dealing with it
        return pydicom.Sequence([tags])

    MANUFACTURER_HANDLERS = {
        "SIEMENS": handle_siemens,
        "GE MEDICAL SYSTEMS": handle_ge,
    }
