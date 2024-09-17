"""Defines the ExcelClient class for communication with the Microsoft Graph API for Excel."""
from time import time
from typing import Any
from urllib.parse import urljoin

import requests

import orthanc  #type: ignore # pylint: disable=import-error
from config import Config

class ExcelClient:
    """Handles communication with the Microsoft Graph API for Excel."""
    def __init__(self, config: Config):
        """Initialize the ExcelClient with the given configuration."""
        microsoft_config = config.get_microsoft_config()
        self.token_url = urljoin(
            microsoft_config['idp_url'],
            microsoft_config['tenant_id'] + microsoft_config['token_endpoint']
        )
        self.token_body = {
            'grant_type': microsoft_config['grant_type'],
            'scope': microsoft_config['scope'],
            'client_id': microsoft_config['client_id'],
            'client_secret': microsoft_config['client_secret'],
            'username': microsoft_config['username'],
            'password': microsoft_config['password']
        }
        self.graph_root_url = microsoft_config['graph_root_url']
        self.access_token = None
        self.token_expires_at = None
        self.cache = {}
        for sheet in config.get_sheets():
            self.cache.update({sheet['drive_id']+sheet['file_id']+sheet['worksheet_id']: {}})

    def get_access_token(self) -> str:
        """Get an access token from the Microsoft Graph API."""
        if (self.access_token is None or
            self.token_expires_at is None or
            self.token_expires_at <= time()):
            response = requests.post(self.token_url, data=self.token_body, timeout=30).json()
            self.access_token = response["access_token"]
            self.token_expires_at = time() + response["expires_in"] - 60
        return self.access_token

    def _excel_request(self, url: str) -> Any:
        """Make a request to the Microsoft Graph API for Excel."""
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.get_access_token()}"},
            timeout=20
        ).json()
        try:
            patient_id = response["text"]
            while isinstance(patient_id, list):
                patient_id = patient_id[0]
        except KeyError:
            orthanc.LogError(f"Failed to retrieve patient ID: {response}")
            return None
        return patient_id


    def get_patient_id(self, key: str, sheet: dict[str, Any], alt_keys = None) -> str:
        """Get the patient ID from the specified sheet using the provided key."""
        if alt_keys is None:
            alt_keys = []

        cache_key = sheet['drive_id']+sheet['file_id']+sheet['worksheet_id']
        if key in self.cache[cache_key]:
            return self.cache[cache_key][key]
        for alt_key in alt_keys:
            if alt_key in self.cache[cache_key]:
                return self.cache[cache_key][alt_key]

        drive_id = sheet['drive_id']
        file_id = sheet['file_id']
        worksheet_id = sheet['worksheet_id']
        mapper_url = urljoin(self.graph_root_url, f"/v1.0/drives/{drive_id}/items/{file_id}/workbook/worksheets/{worksheet_id}/range(address='_{key}')")  # pylint: disable=line-too-long
        patient_id = self._excel_request(mapper_url)

        if patient_id is not None and patient_id != key:
            self.cache[cache_key].update({key: patient_id})
            return patient_id
        for alt_key in alt_keys:
            mapper_url = urljoin(self.graph_root_url, f"/v1.0/drives/{drive_id}/items/{file_id}/workbook/worksheets/{worksheet_id}/range(address='_{alt_key}')")  # pylint: disable=line-too-long
            patient_id = self._excel_request(mapper_url)
            if alt_key == patient_id:
                orthanc.LogError("Alli")
                continue
            self.cache[cache_key].update({key: patient_id})
            orthanc.LogError("Alla")
            return patient_id

        orthanc.LogError(f"Failed to retrieve patient ID for key: {key}")
        return key
