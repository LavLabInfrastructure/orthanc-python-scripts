import orthanc
from config import Config
from excel import ExcelClient
from processor import DicomProcessor
from callback import OrthancCallbackHandler

def main():
    config = Config()
    excel_client = ExcelClient(config)
    dicom_processor = DicomProcessor(config)
    callback_handler = OrthancCallbackHandler(config, dicom_processor, excel_client)
    
    orthanc.RegisterOnStoredInstanceCallback(callback_handler.on_stored_instance)
    orthanc.RegisterRestCallback('/xnat-sync/series', callback_handler.on_series_sync_call)
    orthanc.RegisterRestCallback('/xnat-sync/all', callback_handler.on_all_sync_call)

if __name__ == "__main__":
    main()