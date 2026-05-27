"""ResUNet++ architecture with attention and squeeze-excitation blocks."""

from typing import List
import torch
import torch.nn as nn

from core.modules import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


class ResUnetPlusPlus(nn.Module):
    """ResUNet++ for semantic segmentation.
    
    Enhanced ResUNet with:
    - Squeeze-and-Excitation blocks for channel attention
    - Atrous Spatial Pyramid Pooling (ASPP) for multi-scale features
    - Attention gates for skip connections
    
    Args:
        channel: Number of input channels
        filters: List of filter sizes for each level [f1, f2, f3, f4, f5]
    
    Reference:
        Jha et al., "ResUNet++: An Advanced Architecture for Medical Image
        Segmentation", ISM 2019
    """
    
    def __init__(self, channel: int, filters: List[int] = None) -> None:
        super().__init__()
        
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        
        # Encoder with Squeeze-Excitation
        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])
        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])
        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)
        
        # Bridge with ASPP
        self.aspp_bridge = ASPP(filters[3], filters[4])
        
        # Decoder with Attention gates
        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)
        
        # Output with ASPP
        self.aspp_out = ASPP(filters[1], filters[0])
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
        # Encoder path with Squeeze-Excitation
        enc1 = self.input_layer(x) + self.input_skip(x)
        
        enc2 = self.squeeze_excite1(enc1)
        enc2 = self.residual_conv1(enc2)
        
        enc3 = self.squeeze_excite2(enc2)
        enc3 = self.residual_conv2(enc3)
        
        enc4 = self.squeeze_excite3(enc3)
        enc4 = self.residual_conv3(enc4)
        
        # Bridge with ASPP
        bridge = self.aspp_bridge(enc4)
        
        # Decoder path with attention-gated skip connections
        dec1 = self.attn1(enc3, bridge)
        dec1 = self.upsample1(dec1)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec1 = self.up_residual_conv1(dec1)
        
        dec2 = self.attn2(enc2, dec1)
        dec2 = self.upsample2(dec2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.up_residual_conv2(dec2)
        
        dec3 = self.attn3(enc1, dec2)
        dec3 = self.upsample3(dec3)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec3 = self.up_residual_conv3(dec3)
        
        # Output with ASPP
        features = self.aspp_out(dec3)
        out = self.output_layer(features)
        
        return out
