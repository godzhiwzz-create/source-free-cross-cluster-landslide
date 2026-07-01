"""3D U-Net backbone used in the Sen12Landslides experiments."""

import torch
from torch import nn
import torch.nn.functional as F


def _conv_block(in_dim: int, middle_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )


def _center_in(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )


def _center_out(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(
            in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
    )


def _up_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )


class UNet3DPaper(nn.Module):
    """Paper backbone accepting tensors shaped ``(B, T, C, H, W)``."""

    def __init__(
        self,
        in_channels: int = 11,
        num_classes: int = 1,
        output_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        features = 16
        self.output_size = output_size
        self.en3 = _conv_block(in_channels, features * 4, features * 4)
        self.pool_3 = nn.MaxPool3d(2)
        self.en4 = _conv_block(features * 4, features * 8, features * 8)
        self.pool_4 = nn.MaxPool3d(2)
        self.center_in = _center_in(features * 8, features * 16)
        self.center_out = _center_out(features * 16, features * 8)
        self.dc4 = _conv_block(features * 16, features * 8, features * 8)
        self.trans3 = _up_block(features * 8, features * 4)
        self.dc3 = _conv_block(features * 8, features * 4, features * 2)
        self.final = nn.Conv3d(features * 2, num_classes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        en3 = self.en3(x)
        en4 = self.en4(self.pool_3(en3))
        center = self.center_out(self.center_in(self.pool_4(en4)))
        center = F.interpolate(
            center, size=en4.shape[2:], mode="trilinear", align_corners=True
        )
        decoder = self.dc4(torch.cat((center, en4), dim=1))
        decoder = self.trans3(decoder)
        decoder = F.interpolate(
            decoder, size=en3.shape[2:], mode="trilinear", align_corners=True
        )
        decoder = self.dropout(self.dc3(torch.cat((decoder, en3), dim=1)))
        output = self.temporal_pool(self.final(decoder)).squeeze(2)
        if self.output_size is not None and output.shape[-2:] != (
            self.output_size,
            self.output_size,
        ):
            output = F.interpolate(
                output,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=True,
            )
        return output
