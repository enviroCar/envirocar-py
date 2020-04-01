import unittest

from enpyrocar.client.api.track_api import _parse_track_df, _parse_tracks_list_df

class TestDecoding(unittest.TestCase):

    def setUp(self):
        with open('tests/files/tracks.json', 'r') as myfile:
            self.tracks_list_json = myfile.read()
        
        self.tracks = []
        with open('tests/files/track_1.json', 'r') as track1:
            self.tracks.append(track1.read())
        with open('tests/files/track_2.json', 'r') as track2:
            self.tracks.append(track2.read())

    def test_tracks_list_decoding(self):
        df = _parse_tracks_list_df(self.tracks_list_json)
        self.assertEqual(len(df.index), 100)

    def test_tracks_decoding(self):
        df = _parse_track_df(self.tracks)
        self.assertEqual(len(df['track.id'].unique()), 2)

if __name__ == '__main__':
    unittest.main()