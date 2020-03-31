from ..request_param import RequestParam, BboxSelector, TimeSelector
from ..download_client import DownloadClient

class TrackAPI():
    TRACKS_ENDPOINT = "tracks"
    TRACK_ENDPOINT = "tracks/{}"
    USERTRACKS_ENDPOINT = "users/{}/tracks"
    
    def __init__(self, api_client=None):
        self.api_client = api_client or DownloadClient()

    def get_tracks(self, username=None, bbox:BboxSelector=None, time_interval:TimeSelector=None, num_results=100, page_limit=100):
        
        if username: 
            path = self.USERTRACKS_ENDPOINT.format(username)
        else:
            path = self.TRACKS_ENDPOINT

        return self.api_client.download(RequestParam(path=path))

    def get_track(self, track_id: str):
        pass

        