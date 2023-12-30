import unittest

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import MFCC as MFCCTorch
from kymatio.torch import Scattering1D

from src.main.data.bird import BIRDRoomDimDataset
from src.main.data.features import MelSpec, MFCC, Scat1D


print("Setting up static variables...")
# Get sample data ##############################################################
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
batch_size = 2

ROOT = "C:\\Users\\aidan\\source\\repos\\RIR-encoding"
bird_rd_f01 = BIRDRoomDimDataset(root=ROOT, folder_in_archive='Bird', folds=[1])
x1, _ = bird_rd_f01[0]
x2, _ = bird_rd_f01[1]
X = torch.stack((x1, x2))
X = X.to(device)

# Build AcousticFeature instances ##############################################
sr = 16000
N = sr

# MFCC parameters
n_mfcc = 40
log_mels = True

# mel-spectrogram kwargs
n_fft = 255
win_length = 128
hop_length = 64
melkwargs={'n_fft': n_fft, 'win_length': win_length, 'hop_length': hop_length}

# scattering transform parameters
J = 8
Q1 = 8
Q2 = 1
log2_T = J
T = 2 ** log2_T

# AcousticFeature objects
melspec = MelSpec(sr=sr,
                  batch_size=batch_size,
                  device=device,
                  n_fft=n_fft,
                  win_length=win_length,
                  hop_length=hop_length)
mfcc = MFCC(sr=sr,
            batch_size=batch_size,
            device=device,
            n_mfcc=n_mfcc,
            log_mels=log_mels,
            melkwargs=melkwargs)
st = Scat1D(shape=N,
            sr=sr,
            batch_size=batch_size,
            device=device,
            J=J,
            Q=(Q1, Q2),
            T=T)

# root comparisons
melspec_torch = MelSpectrogram(sample_rate=sr,
                               n_fft=n_fft,
                               win_length=win_length,
                               hop_length=hop_length)
melspec_torch.to(device)
mfcc_torch = MFCCTorch(sample_rate=sr,
                       n_mfcc=n_mfcc,
                       log_mels=log_mels,
                       melkwargs=melkwargs)
mfcc_torch.to(device)
st_kymatio = Scattering1D(shape=N,
                          J=J,
                          Q=(Q1, Q2),
                          T=T)
st_kymatio.to(device)

################################################################################
print("Setup complete.")


class TestMelSpec(unittest.TestCase):
    def test_get_sr(self):
        self.assertEqual(melspec.sr, sr)
    
    def test_get_batch_size(self):
        self.assertEqual(melspec.batch_size, batch_size)

    def test_get_transform(self):
        self.assertEqual(repr(melspec.transform), repr(melspec_torch))
    
    def test_compute_features(self):
        fts = melspec.compute_features(X)
        expected_fts = melspec_torch(X)
        self.assertEqual(fts.tolist(), expected_fts.tolist())


class TestMFCC(unittest.TestCase):
    def test_get_sr(self):
        self.assertEqual(mfcc.sr, sr)

    def test_get_batch_size(self):
        self.assertEqual(mfcc.batch_size, batch_size)

    def test_get_transform(self):
        self.assertEqual(repr(mfcc.transform), repr(mfcc_torch))

    def test_compute_features(self):
        fts = mfcc.compute_features(X)
        fts_expected = mfcc_torch(X)
        self.assertEqual(fts.tolist(), fts_expected.tolist())


class TestScat1D(unittest.TestCase):
    def test_get_sr(self):
        self.assertEqual(st.sr, sr)

    def test_get_batch_size(self):
        self.assertEqual(st.batch_size, batch_size)

    def test_get_transform(self):
        self.assertEqual(repr(st.transform), 'Scattering1D()')

    def test_output_size(self):
        print(f"output size: {st_kymatio.output_size(detail=True)}")
        self.assertEqual(st.transform.output_size(detail=True),
                         st_kymatio.output_size(detail=True))

    def test_compute_features(self):
        fts = st.compute_features(X)
        fts_expected = st_kymatio(X)
        self.assertEqual(fts.tolist(), fts_expected.tolist())


if __name__ == '__main__':
    unittest.main()