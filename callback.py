import io
import json
import threading
from typing import Union
from concurrent.futures import ThreadPoolExecutor

import pydicom
import pydicom.errors
from pynetdicom import AE, StoragePresentationContexts
from pynetdicom.association import Association

import orthanc
from config import Config
from processor import DicomProcessor
from excel import ExcelClient

class OrthancCallbackHandler:
    def __init__(self, config: Config, dicom_processor: DicomProcessor, excel_client: ExcelClient):
        """Initialize the OrthancCallbackHandler with the given configuration, DICOM processor, and Excel client."""
        orthanc_config = config.get_orthanc_config()
        self.xnat_ip = orthanc_config['xnat_ip']
        self.xnat_port = orthanc_config['xnat_port']
        self.ae_title = orthanc_config['ae_title']
        self.xnat_ae_title = orthanc_config['xnat_ae_title']
        self.dicom_processor = dicom_processor
        self.excel_client = excel_client

    def send_to_xnat(self, ds: pydicom.Dataset) -> None:
        """Send the DICOM dataset to XNAT using C-STORE."""
        ae = AE(self.ae_title)
        ae.requested_contexts = StoragePresentationContexts
        assoc: Association = ae.associate(self.xnat_ip, self.xnat_port, ae_title=self.xnat_ae_title)
        
        if assoc.is_established:
            status = assoc.send_c_store(ds)
            assoc.release()
            if status:
                orthanc.LogInfo(f'Successfully sent DICOM file to XNAT')
            else:
                orthanc.LogError(f'Failed to send DICOM file to XNAT')
        else:
            orthanc.LogError(f'Failed to associate with XNAT')

    def process_instance(self, instance: Union[orthanc.DicomInstance, str]) -> None:
        """Process a new DICOM instance by de-identifying and sending it to XNAT."""
        orthanc.LogError(f"Processing instance {instance}")
            
        dcm_bytes = None
        if isinstance(instance, str):
            dcm_bytes = orthanc.GetDicomForInstance(instance)
        else:
            dcm_bytes = instance.GetInstanceData()
        try:
            ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
            ds.filename = None
        except pydicom.errors.InvalidDicomError:
            orthanc.LogError(f'Invalid DICOM instance: {instance}')
            return

        if ds is None:
            orthanc.LogError(f'Could not get DICOM dataset for instance {instance}')
            return
            
        sheet = self.dicom_processor.match_dicom_to_sheet(ds)
        key = sheet["format"].format(ds.get('PatientID'))
        new_patient_id = self.excel_client.get_patient_id(key, sheet)
        if key == new_patient_id:
            orthanc.LogError(f"Could not replace identifier for id: {ds.get('PatientID')}")
        
        deidentified_ds = self.dicom_processor.deidentify_dicom(ds, new_patient_id)
        orthanc.LogInfo(f'Re-identified DICOM with new patient ID: {new_patient_id}')
        self.send_to_xnat(deidentified_ds)

    def on_stored_instance(self, instance_id: str, *args, **kwargs) -> None:
        """Callback function triggered when a new instance is stored in Orthanc."""
        orthanc.LogInfo(f'Processing new instance: {instance_id}')
        threading.Thread(target=self.process_instance, args=(instance_id,)).start()
        # self.process_instance(instance_id)

    def on_series_sync_call(self, output:orthanc.RestOutput, uri, **request):
        orthanc.LogError(f'Syncing series: {request}')
        series_id = request.get('get', {}).get('id')
        if series_id is None:
            return output.SendHttpStatusCode(400)
        try: 
            series_response: dict = json.loads(orthanc.RestApiGet(f'/series/{series_id}'))
        except orthanc.OrthancException:
            return output.SendHttpStatusCode(404)
        except json.JSONDecodeError:
            return output.SendHttpStatusCode(500)
        
        instances = series_response.get('instances', [])
        with ThreadPoolExecutor(4) as tpe:
            tpe.map(self.process_instance, instances)
        return output.SendHttpStatusCode(200)

    def on_all_sync_call(self, output, uri, **request):
        instances = json.loads(orthanc.RestApiGet('/instances'))
        with ThreadPoolExecutor(4) as tpe:
            tpe.map(self.process_instance, instances)
        return output.SendHttpStatusCode(200)