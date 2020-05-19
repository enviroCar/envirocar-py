from scipy import interpolate
import pandas as pd
import numpy as np
import datetime


class Preprocessing():
    def __init__(self):
       print("Hello from preprocessing!") # do we need anything here?

    def interpolate(self, points):
        """ Creates a trajectory from point data
        
        Keyword Arguments:
            points {GeoDataFrame} -- A GeoDataFrame containing the track points
        
        Returns:
            new_points -- An interpolated trajectory, hopefully
        """

        # to have flat attributes
        points['lat'] = points['geometry'].apply(lambda coord: coord.y)
        points['lng'] = points['geometry'].apply(lambda coord: coord.x)
        points_df = pd.DataFrame(points)

        # removing duplicates because interpolation won't work otherwise
        points_df_cleaned = points_df.drop_duplicates(['lat', 'lng'], keep='last')

        # constructing input arrays
        input_coords_y = np.array(points_df_cleaned.lat.values.tolist())
        input_coords_x = np.array(points_df_cleaned.lng.values.tolist())

        # Find the B-spline representation of the curve
        # tck (t,c,k): is a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline.
        # u: is an array of the values of the parameter.
        tck, u = interpolate.splprep([input_coords_x, input_coords_y], s = 0)
        new_points = interpolate.splev(np.linspace(0, 1, 200), tck) # interpolating 200 points

        # TODO interpolate by timestep 
        # TODO interpolate measurements too

        return new_points