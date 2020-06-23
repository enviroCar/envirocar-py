from math import sin, cos, atan2, radians, sqrt
from shapely.geometry import Point

R_EARTH = 6371000  # radius of earth in meters

def measure_distance_spherical(point1, point2):
    """Return spherical distance between two shapely Points as a float."""
    if (type(point1) != Point) or (type(point2) != Point):
        raise TypeError("Only Points are supported as arguments, got {} and {}".format(point1, point2))
    lon1 = float(point1.x)
    lon2 = float(point2.x)
    lat1 = float(point1.y)
    lat2 = float(point2.y)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    a = sin(delta_lat/2) * sin(delta_lat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(delta_lon/2) * sin(delta_lon/2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    dist = R_EARTH * c
    return dist


def measure_distance_euclidean(point1, point2):
    """Return euclidean distance between two shapely Points as float."""
    if (not isinstance(point1, Point)) or (not isinstance(point2, Point)):
        raise TypeError("Only Points are supported as arguments, got {} and {}".format(point1, point2))
    return point1.distance(point2)