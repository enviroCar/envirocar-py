import pandas as pd
# import geopandas as gpd


class TrackConverter():

    """Handles the envirocar Tracks"""

    def __init__(self):
        print("Initializing TrackConverter class")
        # self.track = track
        # self.crs = track.crs

    """ Returns a geoDataFrame object with the movingpandas plain format"""

    def to_movingpandas(self, track):

        # gdf = self.track.copy()
        gdf = track
        gdf = gdf.reindex(columns=(['geometry'] + list([a for a in sorted(
            gdf.columns) if a != 'geometry'])), copy=True)
        gdf['time'] = gdf['time'].astype('datetime64[ns]')
        gdf.set_index('time', inplace=True)
        gdf.index.rename('t', inplace=True)
        return (gdf)

    """ Returns a dataFrame object with the scikitmobility plain format"""

    def to_scikitmobility(self):
        gdf = self.track.copy()
        gdf['lat'] = gdf.geometry.x
        gdf['lng'] = gdf.geometry.y
        gdf.rename(columns=({"time": "datetime", 'sensor.id': 'uid',
                   'track.id': 'tid'}), inplace=True)
        gdf['datetime'] = gdf['datetime'].astype('datetime64[ns]')
        gdf['tid'] = gdf['tid'].astype(str)
        gdf['uid'] = gdf['uid'].astype(str)
        columns = ['uid', 'tid', 'lat', 'lng', 'datetime']
        gdf = gdf.reindex(columns=(columns + list([a for a in sorted(
            gdf.columns) if a not in columns])), copy=True)
        df = pd.DataFrame(gdf)
        return(df)
