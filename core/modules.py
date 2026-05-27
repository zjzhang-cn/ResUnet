"""Core building blocks for ResUNet architectures."""

from typing import List, Optional
import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    """Residual convolutional block with skip connection.
    
    Args:
        input_dim: Number of input channels
        output_dim: Number of output channels
        stride: Stride for the first convolution
        padding: Padding for the first convolution
    """
    
    def __init__(self, input_dim: int, output_dim: int, stride: int, padding: int) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, output_dim, H', W')
        """
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    """Upsampling block using transposed convolution.
    
    Args:
        input_dim: Number of input channels
        output_dim: Number of output channels
        kernel: Kernel size for transposed convolution
        stride: Stride for transposed convolution
    """
    
    def __init__(self, input_dim: int, output_dim: int, kernel: int, stride: int) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, input_dim, H, W)
            
        Returns:
            Upsampled tensor of shape (B, output_dim, H*stride, W*stride)
        """
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    
    Args:
        channel: Number of input channels
        reduction: Reduction ratio for the bottleneck
    
    Reference:
        Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying channel attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attention-weighted tensor of shape (B, C, H, W)
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.
    
    Args:
        in_dims: Number of input channels
        out_dims: Number of output channels
        rate: List of dilation rates for parallel atrous convolutions
    
    Reference:
        Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation", 2017
    """
    
    def __init__(self, in_dims: int, out_dims: int, rate: Optional[List[int]] = None) -> None:
        super().__init__()
        
        if rate is None:
            rate = [6, 12, 18]

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with parallel atrous convolutions.
        
        Args:
            x: Input tensor of shape (B, in_dims, H, W)
            
        Returns:
            Output tensor of shape (B, out_dims, H, W)
        """
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Upsample_(nn.Module):
    """Upsampling block using bilinear interpolation.
    
    Args:
        scale: Upsampling scale factor
    """
    
    def __init__(self, scale: int = 2) -> None:
        super().__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale, align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Upsampled tensor of shape (B, C, H*scale, W*scale)
        """
        return self.upsample(x)


class AttentionBlock(nn.Module):
    """Attention block for skip connections in U-Net architectures.
    
    Args:
        input_encoder: Number of channels from encoder path
        input_decoder: Number of channels from decoder path
        output_dim: Number of output channels
    """
    
    def __init__(self, input_encoder: int, input_decoder: int, output_dim: int) -> None:
        super().__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass computing attention weights.
        
        Args:
            x1: Encoder features of shape (B, input_encoder, H, W)
            x2: Decoder features of shape (B, input_decoder, H', W')
            
        Returns:
            Attention-weighted decoder features
        """
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2
