import requests

from .request_param import RequestParam
from ..exceptions import HttpFailedException


def handle_error_status(http_download):
    def decorate(self, request: RequestParam):
        try:
            return http_download(self, request)
        except requests.HTTPError as exception:
            if exception.response.status_code < requests.status_codes.codes.INTERNAL_SERVER_ERROR:
                message = "Failed to download from: {}".format(request.path)
                raise HttpFailedException(message) from exception
            raise exception from exception
    return decorate
