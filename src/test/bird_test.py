import unittest

from src.main.data.bird import BIRD


ROOT = "C:\\Users\\aidan\\source\\repos\\RIR-scattering"
bird_f01 = BIRD(root=ROOT, folder_in_archive='Bird', folds=[1])

class TestBIRD(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(bird_f01), 10000)

    def test_getitem(self):
        metadata_keys = ['L', 'alpha', 'c', 'mics', 'srcs']
        rir, meta = bird_f01[0]
        self.assertEqual(rir.shape[0], 8)
        self.assertEqual(rir.shape[1], 16000)
        self.assertEqual(list(meta.keys()), metadata_keys)


if __name__ == '__main__':
    unittest.main()
