"""Defines the ExcelClient class for communication with the Microsoft Graph API for Excel."""

from time import time
from typing import Any, Optional
from urllib.parse import urljoin

import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import orthanc  # type: ignore # pylint: disable=import-error
from config import Config


class ExcelClient:
    """Handles communication with the Microsoft Graph API for Excel."""

    def __init__(self, config: Config):
        """Initialize the ExcelClient with the given configuration."""
        microsoft_config = config.get_microsoft_config()
        self.token_url = urljoin(
            microsoft_config["idp_url"],
            microsoft_config["tenant_id"] + microsoft_config["token_endpoint"],
        )
        self.token_body = {
            "grant_type": microsoft_config["grant_type"],
            "scope": microsoft_config["scope"],
            "client_id": microsoft_config["client_id"],
            "client_secret": microsoft_config["client_secret"],
            "username": microsoft_config["username"],
            "password": microsoft_config["password"],
        }
        self.graph_root_url = microsoft_config["graph_root_url"]
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[float] = None

        # Thread-safe cache per sheet
        self.cache: dict[str, dict[str, str]] = {}
        for sheet in config.get_sheets():
            self.cache.update(
                {sheet["drive_id"] + sheet["file_id"] + sheet["worksheet_id"]: {}}
            )

        # HTTP session with retries/backoff
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._lock = threading.RLock()

    def get_access_token(self) -> str:
        """Get an access token from the Microsoft Graph API, refreshing if needed."""
        with self._lock:
            if (
                self.access_token is None
                or self.token_expires_at is None
                or self.token_expires_at <= time()
            ):
                try:
                    resp = self.session.post(
                        self.token_url, data=self.token_body, timeout=30
                    )
                    if resp.status_code != 200:
                        orthanc.LogWarning(
                            f"Token request failed with status {resp.status_code}: {resp.text[:256]}"
                        )
                        # Best-effort: clear token so next call can retry
                        self.access_token = None
                        self.token_expires_at = None
                        return ""
                    response = resp.json()
                    self.access_token = response.get("access_token", "")
                    expires_in = response.get("expires_in", 3600)
                    self.token_expires_at = time() + float(expires_in) - 60
                except Exception as exc:  # pylint: disable=broad-except
                    orthanc.LogWarning(f"Exception while obtaining access token: {exc}")
                    self.access_token = None
                    self.token_expires_at = None
                    return ""
            return self.access_token or ""

    def _get(self, url: str) -> Optional[dict]:
        """Perform a GET with bearer auth, refresh token on 401, and return JSON or None."""
        token = self.get_access_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            resp = self.session.get(url, headers=headers, timeout=30)
            if resp.status_code == 401:
                # Refresh token and retry once
                with self._lock:
                    self.access_token = None
                    self.token_expires_at = None
                token = self.get_access_token()
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                resp = self.session.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                # Log truncated body to avoid flooding logs
                orthanc.LogWarning(
                    f"Excel request failed: {resp.status_code} {resp.reason} for {url}"
                )
                return None
            try:
                return resp.json()
            except Exception as exc:  # pylint: disable=broad-except
                orthanc.LogWarning(f"Failed to decode JSON from Excel response: {exc}")
                return None
        except Exception as exc:  # pylint: disable=broad-except
            orthanc.LogWarning(f"HTTP error while contacting Excel API: {exc}")
            return None

    def _excel_request(self, url: str) -> Optional[str]:
        """Make a request to the Microsoft Graph API for Excel and extract the first scalar cell as text."""
        response = self._get(url)
        if response is None:
            return None

        # Prefer 'text', fallback to 'values'
        cell = None
        if isinstance(response, dict):
            if "text" in response:
                cell = response["text"]
            elif "values" in response:
                cell = response["values"]
            elif "error" in response:
                orthanc.LogWarning(f"Excel API error: {response.get('error')}")
                return None
        # Flatten list-of-lists to a single scalar
        while isinstance(cell, list) and len(cell) > 0:
            cell = cell[0]
        if cell is None:
            return None
        if isinstance(cell, (int, float)):
            return str(cell)
        if isinstance(cell, str):
            return cell
        try:
            return str(cell)
        except Exception:  # pylint: disable=broad-except
            return None

    def get_patient_id(
        self, key: str, sheet: dict[str, Any], alt_keys: Optional[list[str]] = None
    ) -> str:
        """Get the patient ID from the specified sheet using the provided key, with alt keys as fallbacks.
        Uses an in-memory cache and handles API errors gracefully. Returns the input key if no mapping found.
        """
        alt_keys = alt_keys or []
        cache_key = sheet["drive_id"] + sheet["file_id"] + sheet["worksheet_id"]

        # Check cache first (thread-safe)
        with self._lock:
            sheet_cache = self.cache.setdefault(cache_key, {})
            for candidate in [key] + list(alt_keys):
                if candidate in sheet_cache:
                    return sheet_cache[candidate]

        drive_id = sheet["drive_id"]
        file_id = sheet["file_id"]
        worksheet_id = sheet["worksheet_id"]

        def mapper_url(name: str) -> str:
            # Name-based addressing: prefix with '_' as in the original implementation
            return urljoin(
                self.graph_root_url,
                f"/v1.0/drives/{drive_id}/items/{file_id}/workbook/worksheets/{worksheet_id}/range(address='_{name}')",
            )

        # Try primary key first
        if isinstance(key, list):
            # Defensive: never allow a list as a key
            for k in key:
                patient_id = self._excel_request(mapper_url(k))
                if patient_id and patient_id != k:
                    with self._lock:
                        self.cache[cache_key][k] = patient_id
                    return patient_id
        else:
            patient_id = self._excel_request(mapper_url(key))
            if patient_id and patient_id != key:
                with self._lock:
                    self.cache[cache_key][key] = patient_id
                    for ak in alt_keys:
                        self.cache[cache_key].setdefault(ak, patient_id)
                return patient_id

        # Try alternate keys (always as individual strings)
        for ak in alt_keys:
            if isinstance(ak, list):
                for k in ak:
                    patient_id = self._excel_request(mapper_url(k))
                    if patient_id and patient_id != k:
                        with self._lock:
                            self.cache[cache_key][k] = patient_id
                        return patient_id
            else:
                patient_id = self._excel_request(mapper_url(ak))
                if patient_id and patient_id != ak:
                    with self._lock:
                        self.cache[cache_key][key] = patient_id
                        self.cache[cache_key][ak] = patient_id
                    return patient_id

        orthanc.LogWarning(f"No mapping found in Excel for key: {key}")
        # Negative cache to prevent repeated lookups for the same missing key within this run
        with self._lock:
            self.cache[cache_key].setdefault(key, key)
        return key
