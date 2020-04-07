import requests

from .request_param import RequestParam
from ..exceptions import HttpFailedException, NotAuthorizedException


def handle_error_status(http_download):
    """Decorating function for handling enviroCar HTTP exceptions
    
    Arguments:
        http_download {function} -- function to wrap
    
    Raises:
        HttpFailedException: wrapper for client exceptions
        exception: other exceptions
    
    Returns:
        function -- decorated function
    """
    def decorate(self, request: RequestParam):
        try:
            return http_download(self, request)
        except requests.HTTPError as exception:
            if exception.response.status_code < requests.codes['internal_server_error']:
                if exception.response.status_code == requests.codes['unauthorized']:
                    message = "Unauthorized to access: {}".format(request.path)
                    raise NotAuthorizedException(message)
                else: 
                    message = "Failed to download from: {}".format(request.path)
                    raise HttpFailedException(message) from exception
            raise exception from exception
    return decorate
