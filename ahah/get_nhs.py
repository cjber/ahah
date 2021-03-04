# <https://digital.nhs.uk/services/organisation-data-service/data-downloads>

# scotland: <https://www.isdscotland.org/Health-Topics/General-Practice/Workforce-and-Practice-Populations/>

import zipfile

import requests

from ahah.utils import Config


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


for nhs in Config.NHS_FILES.values():
    file = Config.RAW_DATA / nhs
    if not file.exists():
        download_url(Config.NHS_URL + nhs, save_path=file)
