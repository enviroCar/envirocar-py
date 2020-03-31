import logging
import requests 
import warnings
import concurrent.futures
from urllib.parse import urljoin

from .client_config import ECConfig
from .request_param import RequestParam
from .utils import handle_error_status
from ..exceptions import HttpFailedException

LOG = logging.getLogger(__name__)

class DownloadClient:

    def __init__(self, *, config=None):
        self.config = config or ECConfig()

    def request(self, download_requests):
        pass

    def download(self, download_requests, max_workers=None):
        
        if isinstance(download_requests, RequestParam):
            download_requests = [download_requests]

        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            download_list = [executor.submit(self._download, request) for request in download_requests]

        result_list = []
        for future in download_list:
            try:
                result_list.append(future.result())
            except HttpFailedException as e:
                warnings.warn(str(e))
                result_list.append(None)

        return result_list

    @handle_error_status
    def _download(self, download_request: RequestParam):
        url = urljoin(self.config.ec_base_url, download_request.path)
        response = requests.request(
            download_request.method,
            url= url,
            headers=download_request.headers,
            params=download_request.params
        )

        response.raise_for_status()
        LOG.info("Successfully downloaded %s", url)
        return response.content
