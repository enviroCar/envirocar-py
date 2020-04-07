import json
import pandas as pd
import geopandas as gpd

from ..request_param import RequestParam, BboxSelector, TimeSelector
from ..download_client import DownloadClient

class TrackAPI():
    """Handles the API access to the enviroCar backend and returns queried results as dataframes"""
    TRACKS_ENDPOINT = "tracks"
    TRACK_ENDPOINT = "tracks/{}"
    USERTRACKS_ENDPOINT = "users/{}/tracks"
    
    def __init__(self, api_client=None):
        self.api_client = api_client or DownloadClient()

    def get_tracks(self, username=None, bbox:BboxSelector=None, time_interval:TimeSelector=None, num_results=10, page_limit=100):
        """Handles queries against the enviroCar api
        
        Keyword Arguments:
            username {str} -- the username to request tracks for (default: {None})
            bbox {BboxSelector} -- bbox query parameter (default: {None})
            time_interval {TimeSelector} -- time interval query parameter (default: {None})
            num_results {int} -- maximum number of tracks to request (default: {10})
            page_limit {int} -- page limit (default: {100})
        
        Returns:
            GeoDataFrame -- A GeoDataFrame containing the tracks matching the request parameters
        """
        path = self._get_path(username=username)

        # creating download_requests
        download_requests = []
        current_results = 0
        current_page = 1
        while current_results < num_results:
            request_params = {
                'limit': page_limit,
                'page': current_page
            }
            if bbox:
                request_params.update(bbox.param)

            request = RequestParam(path=path, params=request_params)
            download_requests.append(request)
            
            current_results += page_limit
            current_page += 1

        # request for /tracks
        tracks_meta_df = self.api_client.download(download_requests, decoder=_parse_tracks_list_df)
        tracks_meta_df = tracks_meta_df[:num_results]

        ids = tracks_meta_df['track.id'].values
        tracks_df = self._get_tracks_by_ids(ids)

        return tracks_df

    def get_track(self, track_id: str):
        return self.api_client.download(
            RequestParam(path=self._get_path(trackid=track_id)), 
            decoder=_parse_track_df)

    def _get_tracks_by_ids(self, ids: [str]):
        download_requests = [RequestParam(path=self._get_path(trackid=id)) for id in ids]
        return self.api_client.download(download_requests, decoder=_parse_track_df)

    def _get_path(self, *, username=None, trackid=None):
        if username is None and trackid is None:
            return self.TRACKS_ENDPOINT
        if username:
            return self.USERTRACKS_ENDPOINT.format(username)
        if trackid:
            return self.TRACK_ENDPOINT.format(trackid)

def _parse_tracks_list_df(tracks_jsons):
    if not isinstance(tracks_jsons, list):
        tracks_jsons = [tracks_jsons]

    tracks_meta_df = pd.DataFrame()
    for tracks_json in tracks_jsons:
        ec_data = json.loads(tracks_json)
        df = pd.json_normalize(ec_data, 'tracks')
        df.rename(columns=__rename_track_columns, inplace=True)
        tracks_meta_df = tracks_meta_df.append(df)

    return tracks_meta_df

def _parse_track_df(track_jsons):
    if not isinstance(track_jsons, list):
        track_jsons = [track_jsons]
    
    tracks_df = gpd.GeoDataFrame()
    for track_json in track_jsons:
        # read properties
        car_df = pd.json_normalize(json.loads(track_json)['properties'])
        car_df.columns = car_df.columns.str.replace('sensor.properties.', 'sensor.')
        car_df.rename(columns=__rename_track_columns, inplace=True)

        # read geojson values
        track_df = gpd.read_file(track_json)
        track_df = track_df.join(pd.json_normalize(track_df['phenomenons'])).drop(['phenomenons'], axis=1)
        
        # combine dataframes
        car_df = pd.concat([car_df]*len(track_df.index), ignore_index=True)
        tracks_df = tracks_df.append(track_df.join(car_df))
        
    return tracks_df

def __rename_track_columns(x):
    if not x.startswith('sensor'):
        return 'track.' + x
    return x