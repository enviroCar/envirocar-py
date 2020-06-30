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
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    @property
    def param(self):
        return { 'after': f'{self.start_time}' } # TODO

class RequestParam:
    def __init__(self, path: str, method="GET", headers=None, params=None):
        self.path = path
        self.method = method
        self.headers = headers or []
        self.params = params or []

class car():
    def __init__(self, mass=1500, car_cross_sectional=2, Air_drag_cofficient=0.3, Calorific_value= 8.8, Idle_power= 2):
        self.m = mass
        self.A = car_cross_sectional
        self.Cw = Air_drag_cofficient
        self.H_g = Calorific_value
        self.P_idle = Idle_power
class road():
    def __init__(self, rolling_resistance_cofficient = 0.02):
        self.Cr = rolling_resistance_cofficient
    
