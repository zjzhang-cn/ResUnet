"""Residual U-Net architecture."""

from typing import List
import torch
import torch.nn as nn

from core.modules import ResidualConv, Upsample


class ResUnet(nn.Module):
    """Residual U-Net for semantic segmentation.
    
    Uses residual blocks in both encoder and decoder paths.
    
    Args:
        channel: Number of input channels
        filters: List of filter sizes for each level [f1, f2, f3, f4]
    
    Reference:
        Zhang et al., "Road Extraction by Deep Residual U-Net", 2018
    """
    
    def __init__(self, channel: int, filters: List[int] = None) -> None:
        super().__init__()
        
        if filters is None:
            filters = [64, 128, 256, 512]

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)

        # Encoder
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        
        # Bridge (bottleneck)
        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)
        
        # Decoder
        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, channel, H, W)
            
        Returns:
            Output segmentation of shape (B, 1, H, W)
        """
        # Encoder path with skip connections
        enc1 = self.input_layer(x) + self.input_skip(x)
        enc2 = self.residual_conv_1(enc1)
        enc3 = self.residual_conv_2(enc2)
        
        # Bridge (bottleneck)
        bridge = self.bridge(enc3)
        
        # Decoder path with skip connections
        dec1 = self.upsample_1(bridge)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec1 = self.up_residual_conv1(dec1)
        
        dec2 = self.upsample_2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.up_residual_conv2(dec2)
        
        dec3 = self.upsample_3(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec3 = self.up_residual_conv3(dec3)
        
        # Output
        out = self.output_layer(dec3)
        
        return out
