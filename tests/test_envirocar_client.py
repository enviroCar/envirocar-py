import unittest

from enpyrocar.client.client_config import ECConfig
from enpyrocar.client.download_client import DownloadClient
from enpyrocar.client.api.track_api import TrackAPI

class TestDownloadClient(unittest.TestCase):

    def setUp(self):
        self.config = ECConfig()
        self.client = DownloadClient(config=self.config)
        self.track_api = TrackAPI(self.client)

    def test_download(self):
        tracks = self.track_api.get_tracks()
        print(tracks)

if __name__ == '__main__':
    unittest.main()