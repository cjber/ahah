# https://digital.nhs.uk/services/organisation-data-service/data-downloads
# scotland: https://www.isdscotland.org/Health-Topics/General-Practice/Workforce-and-Practice-Populations/
import zipfile

import requests

from ahah.utils import Config

GP = "https://files.digital.nhs.uk/assets/ods/current/epraccur.zip"


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


GP_PATH = Config.RAW_DATA / "epraccur.zip"
if not GP_PATH.exists():
    download_url(GP, save_path=GP_PATH)
    with zipfile.ZipFile(GP_PATH) as zip:
        zip.extractall(Config.RAW_DATA)
