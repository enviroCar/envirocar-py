import logging
import requests 
from requests.auth import HTTPBasicAuth

import warnings
import concurrent.futures
import json
from urllib.parse import urljoin

from .client_config import ECConfig
from .request_param import RequestParam
from .utils import handle_error_status
from ..exceptions import HttpFailedException

LOG = logging.getLogger(__name__)

class DownloadClient:

    def __init__(self, *, config=None):
        self.config = config or ECConfig()

    def download(self, download_requests, decoder=None):
        if (isinstance(download_requests, RequestParam)):
            download_requests = [download_requests]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.number_of_processes) as executor:
            download_list = [executor.submit(self._download, request) for request in download_requests]

        result_list = []
        for future in download_list:
            try:
                decoded_data = future.result().decode('utf-8')
                result_list.append(decoded_data)
            except HttpFailedException as e:
                warnings.warn(str(e))
                result_list.append(None)

        if decoder:
            return decoder(result_list)
        return result_list

    @handle_error_status
    def _download(self, download_request: RequestParam):
        url = urljoin(self.config.ec_base_url, download_request.path)
        
        # set BasicAuth parameters
        auth = None
        if self.config.ec_username and self.config.ec_password:
            auth = HTTPBasicAuth(self.config.ec_username, self.config.ec_password)

        response = requests.request(
            download_request.method,
            url= url,
            auth=auth,
            headers=download_request.headers,
            params=download_request.params
        )

        response.raise_for_status()
        LOG.info("Successfully downloaded %s", url)
        return response.content
