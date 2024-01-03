import unittest

from src.main.data.bird import BIRD, BIRDRoomDimDataset, BIRDSingleRIRDataset


ROOT = "C:\\Users\\aidan\\source\\repos\\RIR-encoding"
bird_f01 = BIRD(root=ROOT, folder_in_archive='Bird', folds=[1])
bird_rd_f01 = BIRDRoomDimDataset(root=ROOT, folder_in_archive='Bird', folds=[1])
bird_sin_f01 = BIRDSingleRIRDataset(root=ROOT, folder_in_archive='Bird', folds=[1])


class TestBIRD(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(bird_f01), 10000)

    def test_getitem(self):
        metadata_keys = ['L', 'alpha', 'c', 'mics', 'srcs']
        rir, meta = bird_f01[0]
        self.assertEqual(rir.shape[0], 8)
        self.assertEqual(rir.shape[1], 16000)
        self.assertEqual(list(meta.keys()), metadata_keys)


class TestBIRDRoomDimDataset(unittest.TestCase):
    def test_getitem(self):
        rir, target = bird_rd_f01[0]
        expected_rir, meta = bird_f01[0]
        expected_target = meta['L']
        self.assertEqual(rir.tolist(), expected_rir.tolist())
        self.assertEqual(target, expected_target)


class TestBIRDSingleRIRDataset(unittest.TestCase):
    def test_getitem(self):
        rir, target = bird_sin_f01[0]
        rirs, meta = bird_f01[0]
        expected_rir = rirs[0]
        expected_target = meta['L']
        self.assertEqual(rir.tolist(), expected_rir.tolist())
        for i in range(len(expected_target)):
            self.assertAlmostEqual(target[i], expected_target[i])


if __name__ == '__main__':
    unittest.main()
