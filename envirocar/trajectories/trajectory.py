import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString


class Trajectory():
    def __init__(self, id, df, id_col):
        self.id = id
        self.df = df
        self.id_col = id_col
        if (df.size > 1):
            self.trajectory = df[df[id_col] == id]

    def __str__(self):
        return "Trajectory {1} ({2} to {3}) | Size: {0}".format(
            self.trajectory.geometry.count(), self.id, self.get_start_time(),
            self.get_end_time())

    def get_start_time(self):
        return self.trajectory['time'].min()

    def get_end_time(self):
        return self.trajectory['time'].max()

    def to_linestring(self):
        return self.make_line(self.trajectory)

    def make_line(self, df):
        if df.size > 1:
            return LineString(df['geometry'])
        else:
            raise RuntimeError(
                'Dataframe needs at least two points to make line!')

    def get_position_at(self, t):
        try:
            return self.trajectory.loc[t]['geometry'][0]
        except Exception:
            return self.trajectory.iloc[
                self.trajectory.index.drop_duplicates().get_loc(
                 t, method='nearest')]['geometry']
    
    def get_number_points(self):
        return self.df.shape[0]
    
    def get_longitudes(self):
        return self.df['geometry'].geometry.x
    
    def get_latitudes(self):
        return self.df['geometry'].geometry.y
    
    def get_coordinates(self):
        coordinates_np = np.zeros((self.get_number_points(), 2))
        coordinates_np[:, 0] = self.get_longitudes()
        coordinates_np[:, 1] = self.get_latitudes()
        return coordinates_np
        
        
            
        
        
