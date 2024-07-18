import re
import base64
import hashlib
from typing import Any
from datetime import datetime

import pydicom
from deid.config import DeidRecipe
from deid.dicom.parser import DicomParser
import pydicom.sequence

import orthanc
from config import Config

class DicomProcessor:
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
        parser.define('hash_func_uid', self.deid_hash_uid_func)
        parser.define('remove_day', self.remove_day)
        
        parser.parse(strip_sequences=False, remove_private=True)
        return parser.dicom
    
    def format_xnat_route(self, item, value, field, dicom):
        xnat_project = self.get_project(dicom)
        return f"Project: {xnat_project} Subject: {dicom.PatientID} Session: {dicom.PatientID}_{dicom.StudyDate}_{dicom.Modality}"
    
    def get_project(self, ds: pydicom.Dataset):
        """Use matching sheet to determine XNAT project"""
        sheet = self.match_dicom_to_sheet(ds)
        return sheet["xnat_project"]
    
    def match_dicom_to_sheet(self, ds: pydicom.Dataset) -> tuple[dict[str, Any], str]:
        """Determine the sheet and XNAT project for the given DICOM dataset based on regex matching."""
        for sheet in self.sheets:
            for tag in ds:
                if re.match(sheet['regex'], str(ds[tag].value)):
                    return sheet, sheet['xnat_project']
        orthanc.LogError("No matching sheet found for the given DICOM dataset")
        return self.sheets[0] # Default to the first sheet

    def deid_hash_uid_func(self, item, value, field, dicom) -> str:
        """Performs self.hash to field.element.value"""
        val = field.element.value
        return DicomProcessor.hash_dicom_uid(str(val))
    
    def deid_hash_func(self, item, value, field, dicom) -> str:
        """Performs self.hash to field.element.value"""
        val = field.element.value
        return DicomProcessor.hash(str(val))

    def gather_diffusion_tags(self, item, value, field, dicom) -> pydicom.Dataset:
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
    
    @staticmethod
    def remove_day(item, value, field, dicom) -> str:
        """Removes the day from a DT field in the deid framework"""
        dt = datetime.strptime(field.element.value, '%Y%m%d')
        return dt.strftime("%Y%m01")

    @staticmethod
    def hash_dicom_uid(uid: str, namespace: str = "1.2.840.113619") -> str:
        """
        Hashes a DICOM UID using SHA-256, truncates the hash, encodes it in Base32, and converts it to a valid numeric UID.
        
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

        numeric_uid = ''.join(str(ord(c) - ord('A') + 10) if 'A' <= c <= 'Z' else c for c in base32_hash)

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
    def handle_siemens(ds: pydicom.Dataset) -> pydicom.Sequence:
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