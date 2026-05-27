"""U-Net architecture and building blocks."""

from typing import Optional
import torch
from torch import nn


class EncodingBlock(nn.Module):
    """Convolutional batch norm block with PReLU activation.
    
    Main building block used in the encoding path of U-Net.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        batch_norm: bool = True,
        dropout: bool = False,
    ) -> None:
        """Initialize encoding block.
        
        Args:
            in_size: Number of input channels
            out_size: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size (usually 0 with reflection padding)
            stride: Convolution stride
            dilation: Dilation rate
            batch_norm: Whether to use batch normalization
            dropout: Whether to use dropout
        """
        super().__init__()

        if batch_norm:

            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
            ]

        else:
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
            ]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, in_size, H, W)
            
        Returns:
            Output tensor of shape (B, out_size, H', W')
        """
        return self.encoding_block(x)


class DecodingBlock(nn.Module):
    """Decoding block with upsampling and skip connections.
    
    Args:
        in_size: Number of input channels (from skip + decoder)
        out_size: Number of output channels
        batch_norm: Whether to use batch normalization
        upsampling: If True, use bilinear upsampling; else use transposed conv
    """
    
    def __init__(
        self,
        in_size: int,
        out_size: int,
        batch_norm: bool = False,
        upsampling: bool = True,
    ) -> None:
        super().__init__()

        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = EncodingBlock(in_size, out_size, batch_norm=batch_norm)

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        """Forward pass combining skip connection with decoder features.
        
        Args:
            skip: Skip connection from encoder of shape (B, C1, H, W)
            decoder: Decoder features of shape (B, C2, H', W')
            
        Returns:
            Decoded features of shape (B, out_size, H, W)
        """
        decoder_up = self.up(decoder)
        skip_resized = nn.functional.interpolate(
            skip, size=decoder_up.size()[2:], mode="bilinear", align_corners=False
        )
        return self.conv(torch.cat([skip_resized, decoder_up], dim=1))


class UNet(nn.Module):
    """Standard U-Net architecture for semantic segmentation.
    
    Args:
        num_classes: Number of output classes/channels
    
    Reference:
        Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
        Image Segmentation", MICCAI 2015
    """

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()

        # Encoding path
        self.conv1 = EncodingBlock(3, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = EncodingBlock(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = EncodingBlock(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = EncodingBlock(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.center = EncodingBlock(512, 1024)

        # Decoding path
        self.decode4 = DecodingBlock(1024, 512)
        self.decode3 = DecodingBlock(512, 256)
        self.decode2 = DecodingBlock(256, 128)
        self.decode1 = DecodingBlock(128, 64)

        # Output layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Output segmentation of shape (B, num_classes, H, W)
        """
        # Encoding path
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # center
        center = self.center(maxpool4)

        # decoding
        decode4 = self.decode4(conv4, center)

        decode3 = self.decode3(conv3, decode4)

        decode2 = self.decode2(conv2, decode3)

        decode1 = self.decode1(conv1, decode2)

        # final
        final = nn.functional.upsample(
            self.final(decode1), input.size()[2:], mode="bilinear"
        )

        return final


class UNetSmall(nn.Module):
    """Smaller U-Net architecture with fewer filters.
    
    Args:
        num_classes: Number of output classes/channels
    """

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()

        # Encoding path
        self.conv1 = EncodingBlock(3, 32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = EncodingBlock(32, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = EncodingBlock(64, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = EncodingBlock(128, 256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.center = EncodingBlock(256, 512)

        # Decoding path
        self.decode4 = DecodingBlock(512, 256)
        self.decode3 = DecodingBlock(256, 128)
        self.decode2 = DecodingBlock(128, 64)
        self.decode1 = DecodingBlock(64, 32)

        # Output layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Output segmentation of shape (B, num_classes, H, W)
        """
        # Encoding path
        enc1 = self.conv1(x)
        enc2 = self.conv2(self.maxpool1(enc1))
        enc3 = self.conv3(self.maxpool2(enc2))
        enc4 = self.conv4(self.maxpool3(enc3))
        
        # Bottleneck
        center = self.center(self.maxpool4(enc4))
        
        # Decoding path with skip connections
        dec4 = self.decode4(enc4, center)
        dec3 = self.decode3(enc3, dec4)
        dec2 = self.decode2(enc2, dec3)
        dec1 = self.decode1(enc1, dec2)
        
        # Output layer with upsampling to match input size
        out = self.final(dec1)
        out = nn.functional.interpolate(
            out, size=x.size()[2:], mode="bilinear", align_corners=False
        )
        
        return out
