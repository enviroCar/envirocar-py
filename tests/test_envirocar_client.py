import unittest
import geopandas as gpd

from enpyrocar.client.client_config import ECConfig
from enpyrocar.client.download_client import DownloadClient
from enpyrocar.client.api.track_api import TrackAPI

class TestDownloadClient(unittest.TestCase):

    def setUp(self):
        self.config = ECConfig()
        self.client = DownloadClient(config=self.config)
        self.track_api = TrackAPI(self.client)

    def test_download(self):
        tracks = self.track_api.get_tracks(num_results=4, page_limit=3)
        self.assertEqual(len(tracks['track.id'].unique()), 4)

if __name__ == '__main__':
    unittest.main()