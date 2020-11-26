from envirocar import TrackAPI, DownloadClient, BboxSelector, ECConfig

# create an initial but optional config and an api client
config = ECConfig()
track_api = TrackAPI(api_client=DownloadClient(config=config))
## Mönchengladbach bbox coordinates

'''
bbox = BboxSelector([
    6.3301849365234375, # min_x
    51.13369295212583, # min_y
    6.540985107421875, # max_x
    51.23870648334856  # max_y
])
# issue a query
track_df = track_api.get_tracks(bbox=bbox, num_results=500) # requesting 500 tracks inside the bbox
track_df.head()
'''

'''locally saving the files using feather'''
data_path = 'Data/Monchengladbach_500_complete.feather'
track_df.reset_index(inplace = True)
to_geofeather(track_df, data_path)

#bounding box polygon in geodataframe
from shapely.geometry import Polygon
bboxpoly = Polygon([(6.3301849365234375, 51.13369295212583,), 
                                       (6.540985107421875,51.13369295212583), 
                                       (6.540985107421875, 51.23870648334856 ), 
                                       (6.3301849365234375,51.23870648334856)])
bboxpoly_gdf = gpd.GeoDataFrame([1],geometry=[bboxpoly])

#plotting the tracks and bounding box shows the extending tracks
# base = bboxpoly_gdf.plot(color='white', edgecolor='red')
# track_df.plot(ax=base)

#clipping track_df by bboxpoly_gdf
track_df_copy = track_df.reset_index(drop = True)
clipTracks_df = gpd.clip(track_df_copy, bboxpoly_gdf)

#conversion to feather format
data_path = 'Data/Monchengladbach_500.feather'
clipTracks_df.reset_index(inplace = True)
to_geofeather(clipTracks_df, data_path)