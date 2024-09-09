"""Defines DicomProcessor class for de-identifying DICOM datasets."""
import re
import base64
import hashlib
from typing import Any
from datetime import datetime

import pydicom
import pydicom.sequence
from deid.config import DeidRecipe
from deid.dicom.parser import DicomParser

import orthanc  # pylint: disable=import-error

from config import Config

# pylint: disable=unused-argument
class DicomProcessor:
    """Handles de-identification of DICOM datasets."""
    def __init__(self, config: Config):
        """Initialize the DicomProcessor with the given configuration."""
        orthanc_config = config.get_orthanc_config()
        deid_config = orthanc_config['deid']
        self.recipe = DeidRecipe(deid_config['recipe_paths'], deid_config['use_base_recipe'])
        self.sheets = config.get_sheets()

    def deidentify_dicom(self, ds: pydicom.Dataset, patient_id: str) -> pydicom.Dataset:
        """De-identify the DICOM dataset and return the de-identified dataset."""
        parser = DicomParser(ds, self.recipe)

        parser.define('excel_id', patient_id)
        parser.define('format_xnat_route', self.format_xnat_route)
        parser.define('format_xnat_route_antiphi', self.format_xnat_route_antiphi)
        parser.define('hash_uid_func', self.deid_hash_uid_func)
        parser.define('remove_day', self.remove_day)
        parser.define('hash_func', self.deid_hash_func)
        parser.define('gather_diffusion_tags', self.gather_diffusion_tags)
        parser.define('round_func', self.deid_round_func)

        parser.parse(strip_sequences=False, remove_private=True)
        return parser.dicom

    def format_xnat_route_antiphi(self, item, value, field, dicom):
        """
        Formats the XNAT route and assigns to the given field.
        This variant checks for KO and SR modalities and ConversionType 
        to determine if the session is PHI.
        """

        xnat_project = self.get_project(dicom)
        # logic to remove icky stuff
        modality = dicom.get('Modality')
        if modality == "KO" or modality == "SR":
            sesh = "PROBABLE_PHI"
        elif dicom.get("ConversionType") is not None:
            sesh = "PROBABLE_PHI"
        else:
            sesh = f"{dicom.get('PatientID')}_{dicom.get('StudyDate')}_{dicom.get('Modality')}"

        return f"Project: {xnat_project} Subject: {dicom.get('PatientID')} Session: {sesh}"

    def format_xnat_route(self, item, value, field, dicom):
        """Formats the XNAT route and assigns to the given field."""
        xnat_project = self.get_project(dicom)
        sesh = f"{dicom.get('PatientID')}_{dicom.get('StudyDate')}_{dicom.get('Modality')}"
        return f"Project: {xnat_project} Subject: {dicom.get('PatientID')} Session: {sesh}"

    def get_project(self, ds: pydicom.Dataset):
        """Use matching sheet to determine XNAT project"""
        sheet = self.match_dicom_to_sheet(ds)
        return sheet["xnat_project"]

    def match_dicom_to_sheet(self, ds: pydicom.Dataset) -> tuple[dict[str, Any], str]:
        """Find the sheet and XNAT project for the given DICOM dataset based on regex matching."""
        for sheet in self.sheets:
            for identifier in sheet['identifiers']:
                tag = identifier['tag']
                regex = identifier['regex']
                tag_val = ds.get(tag)
                if tag_val is None:
                    continue
                if not isinstance(tag_val, str):
                    tag_val = tag_val.value
                if re.match(regex, str(tag_val)):
                    return sheet
        orthanc.LogError("No matching sheet found for the given DICOM dataset")
        return self.sheets[0] # Default to the first sheet

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
        manufacturer = dicom.Manufacturer.upper() if 'Manufacturer' in dicom else None
        if not manufacturer:
            raise ValueError("Manufacturer information is missing from the DICOM file")

        handler = self.MANUFACTURER_HANDLERS.get(manufacturer)
        if not handler:
            raise ValueError(f"Unsupported or unknown manufacturer: {manufacturer}")

        # Create MR Diffusion Sequence if it doesn't exist
        if 'MRDiffusionSequence' not in dicom:
            dicom.MRDiffusionSequence = [pydicom.Dataset()]

        # Call the handler function to get DiffusionSequence
        return handler(dicom)

    def deid_round_func(self, item, value, field, dicom) -> str:
        """Performs self.hash to field.element.value"""
        val = str(field.element.value).lower().strip('y')
        return f"{self.round_to_nearest_five(int(val)):03}Y"
    @staticmethod
    def remove_day(item, value, field, dicom) -> str:
        """Removes the day from a DT field in the deid framework"""
        dt = datetime.strptime(field.element.value, '%Y%m%d')
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
        sha256.update(uid.encode('utf-8'))
        hash_bytes = sha256.digest()

        truncated_hash_bytes = hash_bytes[:24]

        base32_hash = base64.b32encode(truncated_hash_bytes).decode('utf-8').rstrip('=')

        numeric_uid = ''.join(str(ord(c) - ord('A') + 10) if 'A' <= c <= 'Z' else c for c in base32_hash)  # pylint: disable=line-too-long

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
        sha256.update(uid.encode('utf-8'))
        return sha256.hexdigest()

    @staticmethod
    def round_to_nearest_five(n):
        """Rounds a number to the nearest multiple of 5."""
        return 5 * round(n / 5)

    @staticmethod
    def handle_siemens(ds: pydicom.Dataset) -> pydicom.Sequence:
        """Handles Siemens specific diffusion tags."""
        tags = pydicom.Dataset()
        #bval
        if (0x0019, 0x100C) in ds:
            value = float(ds[(0x0019, 0x100C)].value)
            tags.add_new((0x0018,0x9087), 'FD', value)

        # directionality
        if (0x0019, 0x100D) in ds:
            value = str(ds[(0x0019, 0x100D)].value)
            tags.add_new((0x0018,0x9075), 'CS', value.upper())

        # gradient direction and sequence (gradient orientation)
        if (0x0019, 0x100E) in ds:
            value = ds[(0x0019, 0x100E)].value
            _tag = pydicom.Dataset()
            _tag.add_new((0x0018,0x9089), 'FD', value)
            tags.add_new((0x0018,0x9076), 'SQ', pydicom.Sequence([_tag]))

        # bmatrix and sequence
        if (0x0019, 0x1027) in ds:
            value = ds[(0x0019, 0x1027)].value
            # should really be 6, hopefully this never hits
            if len(value) is not 6:
                return
            # decode bmatrix into their appropriate tags
            _tags = pydicom.Dataset()
            tag = 0x9602
            for val in value:
                _tags.add_new((0x0018,tag), 'FD', val)
                tag = hex(tag + 0x0001)
            tags.add_new((0x0018,0x9601), 'SQ', pydicom.Sequence([_tags]))

        return pydicom.Sequence([tags])

    @staticmethod
    def handle_ge(ds: pydicom.Dataset) -> None:
        """Handles GE specific diffusion tags."""
        tags = pydicom.Dataset()
        #bval
        if (0x0043,0x1039) in ds:
            value = int(ds[(0x0019, 0x100C)].value[0])
            if value > 10000:
                value = value % 100000 #god GE is so annoying
            tags.add_new((0x0018,0x9087), 'FD', value)

        ## GE DTI SUCKS EVEN MORE! waiting until we need it before dealing with it
        return pydicom.Sequence([tags])

    MANUFACTURER_HANDLERS = {
        'SIEMENS': handle_siemens,
        'GE MEDICAL SYSTEMS': handle_ge,
    }
