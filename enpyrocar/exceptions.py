
class HttpFailedException(Exception):
    """ General exception raised whenever communication failes """

class NotAuthorizedException(HttpFailedException):
    """ Exception raised whenever the user doesn't has the rights to access a certain resource """