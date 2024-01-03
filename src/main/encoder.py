from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0
import lightning as L
from nnAudio.features import MelSpectrogram


class EfficientNet(L.LightningModule):
    """EfficientNetB0 model for encoding mel-spectrograms of audio signals."""
    
    def __init__(self) -> None:
        super().__init__()

        # compute mel-spectrogram using nnAudio (not trainable)
        self.spec_layer = MelSpectrogram(sr=16000,
                                         n_fft=255,
                                         win_length=128,
                                         hop_length=64,
                                         trainable_mel=False,
                                         trainable_STFT=False)
        
        # adapt to efficientnet_b0 input shape (3 channels)
        self.pre_input_layer = nn.Sequential(
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.Conv2d(in_channels=1, out_channels=3,kernel_size=(3,3))
        )

        self.model = efficientnet_b0(num_classes=3)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, input_tensor):
        #batch_size, n_rir, length = input_tensor.shape
        #input_tensor = torch.reshape(input_tensor, (batch_size * n_rir, length))
        x = self.spec_layer(input_tensor)
        x = x.unsqueeze(1)
        x = self.pre_input_layer(x)
        x = self.model(x)
        return x
    
    def mse_loss(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)
    
    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        loss = self._step(train_batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
        loss = self._step(val_batch)
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx) -> STEP_OUTPUT:
        loss = self._step(test_batch)
        self.log('test_loss', loss)

    def _step(self, batch):
        X, y = batch
        y_pred = self.forward(X)
        loss = self.mse_loss(y_pred, y)
        return loss
