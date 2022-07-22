class BboxSelector:
    """ Class for creating bounding box requests """

    def __init__(self, bbox: [float]):
        self.min_x = bbox[0]
        self.min_y = bbox[1]
        self.max_x = bbox[2]
        self.max_y = bbox[3]

    @property
    def lower_left(self):
        return self.min_x, self.min_y

    @property
    def upper_right(self):
        return self.max_x, self.max_y

    @property
    def param(self):
        return { 'bbox': f'{self.min_x},{self.min_y},{self.max_x},{self.max_y}' }

class TimeSelector:
    def __init__(self, start_time=None, end_time=None):
        self.start_time = start_time
        self.end_time = end_time

    @property
    def param(self):
        if self.start_time and self.end_time==None:
            return { 'after': f'{self.start_time}' }
        elif self.start_time==None and self.end_time:
            return { 'before': f'{self.end_time}' }
        elif self.start_time and self.end_time:
            return { 'during': f'{self.start_time},{self.end_time}' }


class RequestParam:
    def __init__(self, path: str, method="GET", headers=None, params=None):
        self.path = path
        self.method = method
        self.headers = headers or []
        self.params = params or []
