import torch
import torch.nn as nn
from typing import List

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, num_layers: int = 3, base_filters: int = 32):
        super().__init__()
        filters: List[int] = [base_filters * (2 ** i) for i in range(num_layers)]
        convs = []
        in_ch = 3
        
        for i, f in enumerate(filters):
            convs.append(nn.Conv2d(in_ch, f, kernel_size=3, padding=1))
            convs.append(nn.BatchNorm2d(f))
            convs.append(nn.ReLU(inplace=True))
            
            # Only add pooling for first few layers to prevent spatial dimensions from becoming zero
            # For CIFAR-10 (32x32), we can safely do at most 5 pooling operations (32/2^5 = 1)
            if i < min(5, num_layers):  # Limit pooling to prevent zero spatial dimensions
                convs.append(nn.MaxPool2d(kernel_size=2))
            
            in_ch = f
            
        self.features = nn.Sequential(*convs)
        
        # Calculate spatial dimensions more carefully
        num_pools = min(5, num_layers)  # Number of actual pooling operations
        spatial = 32 // (2 ** num_pools)
        spatial = max(1, spatial)  # Ensure at least 1x1
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * spatial * spatial, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor):
        maps = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                maps.append(x.detach().cpu())
        return maps
