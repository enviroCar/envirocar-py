
class HttpFailedException(Exception):
    """ General exception raised whenever communication failes """

class NotFoundException(HttpFailedException):
    """ Exception raised whenever a certain resource has not been found """

class NotAuthorizedException(HttpFailedException):
    """ Exception raised whenever the user doesn't has the rights to access a certain resource """

class MailNotConfirmedException(HttpFailedException):
    """ Exception raised whenever the user hasn't confirmed his mail """

