from abc import ABC, abstractmethod

import math
from kymatio.torch import Scattering1D
import torch
import torchaudio.transforms as T


class AcousticFeature(ABC):
    """
    An interface for computing features from raw audio signals.

    Attributes
    ----------
    sr : int
        The sample rate of each audio signal.
    batch_size : int
        The number of audio signals to process at once.
    """
    def __init__(self, sr=16000, batch_size=1, device='cpu') -> None:
        super().__init__()
        self.sr = sr
        self.batch_size = batch_size

    def to_device(self, device='cpu'):
        self.transform = (
            self.transform.cuda() if device == "cuda" else self.transform.cpu()
        )

    @abstractmethod
    def compute_features(self):
        raise NotImplementedError("This method must implement computation of" +
                                  " the features (dependent on subclass).")
    

class MelSpec(AcousticFeature):
    """
    Computes the mel-spectrogram of an audio signal.

    Attributes
    ----------
    sr : int
        The sample rate of each audio signal.
    batch_size : int
        The number of audio signals to process at once.
    device : str
        The device to use for computation (either 'cpu' or 'cuda').
    n_fft : int
        Size of the fast Fourier transform (FFT).
    win_length : int
        Window size for the short-time Fourier transform (STFT).
    hop_length : int
        Length of hop between STFT windows.

    """
    def __init__(self, 
                 sr=16000, 
                 batch_size=1,
                 device='cpu',
                 n_fft=255,
                 win_length=128,
                 hop_length=64) -> None:
        super().__init__(sr, batch_size)
        self.transform = T.MelSpectrogram(sample_rate=sr,
                                          n_fft=n_fft,
                                          win_length=win_length,
                                          hop_length=hop_length)
        self.to_device(device)

    def compute_features(self, X):
        """
        Computes the mel-spectrograms for the given audio signal(s).

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape (batch_size, shape), which are the audio signal(s)
            to compute the 1D scattering transform for.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, n_mels, time) where n_mels is the
            number of mel filterbanks.

        """
        return self.transform(X)
    

class MFCC(AcousticFeature):
    """
    Computes the mel-frequency cepstral coefficients of an audio signal.

    Attributes
    ----------
    sr : int
        The sample rate of each audio signal.
    batch_size : int
        The number of audio signals to process at once.
    device : str
        The device to use for computation (either 'cpu' or 'cuda').
    n_mfcc : int
        Number of coefficients to compute.
    log_mels : bool
        Whether to use log-mel spectrograms instead of db-scaled.
    melkwargs : dict
        Arugments for computing the mel-spectrogram (see n_fft, win_length,
        hop_length from MelSpec).
    global_avg : bool
        Whether or not to compute the global average of each coefficient.

    """
    def __init__(self, 
                 sr=16000, 
                 batch_size=1,
                 device='cpu',
                 n_mfcc=40,
                 log_mels=True,
                 melkwargs={'n_fft': 255,
                            'win_length': 128,
                            'hop_length': 64},
                 global_avg=True) -> None:
        super().__init__(sr, batch_size)
        self.transform = T.MFCC(sample_rate=1600,
                                n_mfcc=n_mfcc,
                                log_mels=log_mels,
                                melkwargs=melkwargs)
        self.to_device(device)
        self.global_avg = global_avg
    
    def compute_features(self, X):
        """
        Computes the MFCCs for the given audio signal(s).

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape (batch_size, shape), which are the audio signal(s)
            to compute the 1D scattering transform for.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, n_mfcc), which are the computed
            MFCCs.

        """
        mfccs = self.transform(X)
        if self.global_avg:
            mfccs = mfccs.mean(dim=-1)
        return mfccs
    

class Scat1D(AcousticFeature):
    """
    Computes the 1D scattering transform of an audio signal.

    Attributes
    ----------
    shape : int
        The length of each audio signal (# of samples).
    sr : int
        The sample rate of each audio signal.
    batch_size : int
        The number of audio signals to process at once.
    device : str
        The device to use for computation (either 'cpu' or 'cuda').
    J : int
        The maximum log-scale of the scattering transform
    Q : int or tuple 
        The number of wavelets for the first and second orders of the
        scattering transform (Q1, Q2). If Q is an int, this corresponds to
        choosing (Q1=Q, Q2=1).
    T : int
        The temporal support of the low-pass filter, controlling amount of
        imposed time-shift invariance. If None, T=2**J.
    global_avg : bool
        Whether or not to compute the global average of each scattering 
        coefficient path when computing features.

    """
    def __init__(self,
                 shape, 
                 sr=16000, 
                 batch_size=1,
                 device='cpu',
                 J=8,
                 Q=(8, 1),
                 T=None,
                 global_avg=True) -> None:
        super().__init__(sr, batch_size)
        self.transform = Scattering1D(shape=shape, J=J, Q=Q, T=T)
        self.to_device(device)
        self.global_avg = global_avg
    
    def compute_features(self, X):
        """
        Computes the 1D scattering transform for the given audio signal(s).

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape (batch_size, shape), which are the audio signal(s)
            to compute the 1D scattering transform for.
        
        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, C), where C is the number of
            scattering coefficients if global_avg=True, or a tensor of shape
            (batch_size, C, L) where L is the length of each coefficient path.

        """
        fts = self.transform(X)
        if self.global_avg:
            fts = fts.mean(dim=-1)
        return fts
