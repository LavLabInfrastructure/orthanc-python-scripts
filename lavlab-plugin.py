"""Entry point for the Orthanc plugin."""
import os
import sys

import orthanc  # pylint: disable=import-error


# add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add the virtual environment to sys.path
def add_venv_to_path():
    """Adds a virtual environment to sys.path if not already in one."""
    # Check if running inside a virtual environment
    if hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Already inside a virtual environment
        orthanc.LogInfo("Running inside a virtual environment")
        return
    # Look for a virtual environment manually
    venv_path = os.environ.get('VIRTUAL_ENV', '/tmp/venv')
    if not os.path.exists(venv_path):
        orthanc.LogError("No virtual environment found")
        return

    orthanc.LogInfo(f"Found virtual environment at {venv_path}")
    # Add the site-packages to sys.path
    site_packages = os.path.join(venv_path, 'lib',
                                    f'python{sys.version_info.major}.{sys.version_info.minor}',
                                    'site-packages')
    if not os.path.exists(site_packages):
        orthanc.LogError(f"Site-packages directory not found in {venv_path}")
    sys.path.append(site_packages)
    orthanc.LogInfo(f"Added {site_packages} to sys.path")
add_venv_to_path()

# need to import after adding the virtual environment to sys.path
# pylint: disable=wrong-import-position
from config import Config
from excel import ExcelClient
from processor import DicomProcessor
from callback import OrthancCallbackHandler

config = Config(os.environ.get('ORTHANC_PYTHON_SCRIPTS_YAML', 'config.yaml'))
excel_client = ExcelClient(config)
dicom_processor = DicomProcessor(config)
callback_handler = OrthancCallbackHandler(config, dicom_processor, excel_client)

orthanc.RegisterOnStoredInstanceCallback(callback_handler.on_stored_instance)
orthanc.RegisterRestCallback('/sync-series', callback_handler.on_series_sync_call)
orthanc.RegisterRestCallback('/sync-all', callback_handler.on_all_sync_call)
