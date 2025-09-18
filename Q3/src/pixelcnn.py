import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class MaskedConv2d(nn.Module):
    """
    Masked Convolution for PixelCNN
    
    Mask Type A: Used for the first layer, masks the center pixel and all pixels to the right/below
    Mask Type B: Used for later layers, allows the center pixel but masks pixels to the right/below
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 mask_type: str, stride: int = 1, padding: int = 0, bias: bool = True):
        super(MaskedConv2d, self).__init__()
        
        assert mask_type in ['A', 'B'], "Mask type must be 'A' or 'B'"
        
        self.mask_type = mask_type
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=bias)
        
        # Create the mask
        self.register_buffer('mask', self._create_mask(kernel_size, in_channels, out_channels))
        
    def _create_mask(self, kernel_size: int, in_channels: int, out_channels: int) -> torch.Tensor:
        """Create the autoregressive mask"""
        mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        
        center = kernel_size // 2
        
        # Mask everything below the center row
        mask[:, :, center + 1:, :] = 0
        
        # Mask everything to the right of center in the center row
        mask[:, :, center, center + 1:] = 0
        
        # For mask type A, also mask the center pixel
        if self.mask_type == 'A':
            mask[:, :, center, center] = 0
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask to weights before convolution
        self.conv.weight.data *= self.mask
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block for PixelCNN with masked convolutions"""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = MaskedConv2d(channels, channels // 2, kernel_size=1, mask_type='B')
        self.conv2 = MaskedConv2d(channels // 2, channels // 2, kernel_size=kernel_size, 
                                 mask_type='B', padding=kernel_size // 2)
        self.conv3 = MaskedConv2d(channels // 2, channels, kernel_size=1, mask_type='B')
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        
        return out + residual


class PixelCNN(nn.Module):
    """
    PixelCNN implementation for autoregressive image generation
    
    Based on "Pixel Recurrent Neural Networks" by van den Oord et al.
    """
    
    def __init__(self, input_channels: int = 3, hidden_channels: int = 128, 
                 num_layers: int = 12, num_classes: int = 256):
        super(PixelCNN, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # First layer with mask type A
        self.first_conv = MaskedConv2d(input_channels, hidden_channels, 
                                      kernel_size=7, mask_type='A', padding=3)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_conv1 = MaskedConv2d(hidden_channels, hidden_channels, 
                                        kernel_size=1, mask_type='B')
        self.output_conv2 = MaskedConv2d(hidden_channels, input_channels * num_classes, 
                                        kernel_size=1, mask_type='B')
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (B, C, H, W) with pixel values [0, 255]
            
        Returns:
            logits: (B, C, num_classes, H, W) logits for each pixel and color channel
        """
        batch_size, channels, height, width = x.shape
        
        # Convert to float and normalize to [0, 1]
        x = x.float() / 255.0
        
        # First convolution
        out = self.relu(self.first_conv(x))
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Output layers
        out = self.relu(self.output_conv1(out))
        out = self.output_conv2(out)
        
        # Reshape to (B, C, num_classes, H, W)
        out = out.view(batch_size, channels, self.num_classes, height, width)
        
        return out
    
    def sample(self, shape: tuple, device: torch.device, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate samples using autoregressive sampling
        
        Args:
            shape: (batch_size, channels, height, width)
            device: Device to generate on
            temperature: Sampling temperature
            
        Returns:
            Generated images (B, C, H, W) with values [0, 255]
        """
        batch_size, channels, height, width = shape
        
        # Initialize with zeros
        samples = torch.zeros(shape, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    for c in range(channels):
                        # Get logits for current pixel
                        logits = self.forward(samples)
                        logits = logits[:, c, :, i, j] / temperature
                        
                        # Sample from categorical distribution
                        probs = F.softmax(logits, dim=-1)
                        pixel_value = torch.multinomial(probs, 1).squeeze(-1)
                        
                        # Update the sample
                        samples[:, c, i, j] = pixel_value
        
        return samples
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'PixelCNN',
            'input_channels': self.input_channels,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


if __name__ == '__main__':
    # Test PixelCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PixelCNN(input_channels=3, hidden_channels=64, num_layers=6).to(device)
    
    # Test forward pass
    batch_size = 4
    x = torch.randint(0, 256, (batch_size, 3, 32, 32), dtype=torch.long).to(device)
    
    print("Testing PixelCNN...")
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input range: [{x.min()}, {x.max()}]")
    
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: (batch_size, channels, num_classes, height, width)")
    
    # Test model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test sampling (small image for speed)
    print(f"\nTesting sampling...")
    sample_shape = (1, 3, 8, 8)
    samples = model.sample(sample_shape, device, temperature=1.0)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample range: [{samples.min()}, {samples.max()}]")