import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from torchvision.utils import make_grid


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(directory: str):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate negative log-likelihood for discrete pixel values
    
    Args:
        logits: (B, C, num_classes, H, W) model logits
        targets: (B, C, H, W) target pixel values [0, 255]
        
    Returns:
        Average negative log-likelihood per pixel
    """
    batch_size, channels, num_classes, height, width = logits.shape
    
    # Reshape for cross-entropy calculation
    logits = logits.permute(0, 2, 1, 3, 4).contiguous()  # (B, num_classes, C, H, W)
    logits = logits.view(batch_size * channels * height * width, num_classes)
    
    targets = targets.view(batch_size * channels * height * width)
    
    # Calculate cross-entropy
    nll = F.cross_entropy(logits, targets, reduction='mean')
    
    return nll.item()


def calculate_bits_per_dimension(nll: float, height: int, width: int, channels: int) -> float:
    """
    Convert negative log-likelihood to bits per dimension
    
    Args:
        nll: Negative log-likelihood (nats)
        height: Image height
        width: Image width
        channels: Number of channels
        
    Returns:
        Bits per dimension
    """
    # Convert from nats to bits and normalize by number of dimensions
    total_dims = height * width * channels
    bits_per_dim = nll / (np.log(2) * total_dims)
    return bits_per_dim


class PixelCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for pixel-level predictions"""
    
    def __init__(self, reduction: str = 'mean'):
        super(PixelCrossEntropyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss
        
        Args:
            logits: (B, C, num_classes, H, W) model logits
            targets: (B, C, H, W) target pixel values [0, 255]
            
        Returns:
            Cross-entropy loss
        """
        batch_size, channels, num_classes, height, width = logits.shape
        
        # Reshape for cross-entropy calculation
        logits = logits.permute(0, 2, 1, 3, 4).contiguous()  # (B, num_classes, C, H, W)
        logits = logits.view(batch_size * channels * height * width, num_classes)
        
        targets = targets.view(batch_size * channels * height * width)
        
        return F.cross_entropy(logits, targets, reduction=self.reduction)


def sample_from_model(model: nn.Module, shape: tuple, device: torch.device, 
                     temperature: float = 1.0, num_samples: int = 8) -> torch.Tensor:
    """
    Generate samples from the model
    
    Args:
        model: Trained model
        shape: (channels, height, width) for single image
        device: Device to generate on
        temperature: Sampling temperature
        num_samples: Number of samples to generate
        
    Returns:
        Generated samples (num_samples, C, H, W)
    """
    model.eval()
    
    full_shape = (num_samples,) + shape
    samples = model.sample(full_shape, device, temperature)
    
    return samples


def visualize_samples(samples: torch.Tensor, save_path: str = None, 
                     title: str = "Generated Samples", nrow: int = 4):
    """
    Visualize generated samples
    
    Args:
        samples: (N, C, H, W) generated samples with values [0, 255]
        save_path: Path to save the visualization
        title: Title for the plot
        nrow: Number of images per row
    """
    # Convert to float and normalize to [0, 1]
    samples_float = samples.float() / 255.0
    
    # Create grid
    grid = make_grid(samples_float, nrow=nrow, padding=2, normalize=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_models(results: Dict[str, Dict], save_path: str = None):
    """
    Compare model performance
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        save_path: Path to save the comparison plot
    """
    models = list(results.keys())
    metrics = ['val_nll', 'val_bpd', 'parameters']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # NLL comparison
    nlls = [results[model]['val_nll'] for model in models]
    axes[0].bar(models, nlls, color=['blue', 'orange', 'green'])
    axes[0].set_title('Validation Negative Log-Likelihood')
    axes[0].set_ylabel('NLL (lower is better)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # BPD comparison
    bpds = [results[model]['val_bpd'] for model in models]
    axes[1].bar(models, bpds, color=['blue', 'orange', 'green'])
    axes[1].set_title('Bits per Dimension')
    axes[1].set_ylabel('BPD (lower is better)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Parameter count comparison
    params = [results[model]['parameters'] / 1e6 for model in models]  # Convert to millions
    axes[2].bar(models, params, color=['blue', 'orange', 'green'])
    axes[2].set_title('Model Size')
    axes[2].set_ylabel('Parameters (millions)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_inception_score(samples: torch.Tensor, batch_size: int = 32, 
                            splits: int = 10) -> tuple:
    """
    Calculate Inception Score (IS) for generated samples
    
    Note: This is a simplified version. For production use, 
    you should use the full Inception v3 network.
    
    Args:
        samples: (N, C, H, W) generated samples [0, 255]
        batch_size: Batch size for processing
        splits: Number of splits for calculating IS
        
    Returns:
        (mean_is, std_is): Mean and standard deviation of IS
    """
    # This is a placeholder implementation
    # In practice, you would use a pre-trained Inception v3 model
    # and calculate the KL divergence between p(y|x) and p(y)
    
    print("Note: This is a placeholder IS implementation.")
    print("For accurate IS calculation, use a pre-trained Inception v3 model.")
    
    # Return dummy values for now
    return 5.0, 0.5


def calculate_fid_score(real_samples: torch.Tensor, fake_samples: torch.Tensor) -> float:
    """
    Calculate Fréchet Inception Distance (FID) between real and fake samples
    
    Note: This is a simplified version. For production use,
    you should use the full Inception v3 network and calculate
    the Fréchet distance between feature distributions.
    
    Args:
        real_samples: (N, C, H, W) real samples [0, 255]
        fake_samples: (N, C, H, W) generated samples [0, 255]
        
    Returns:
        FID score (lower is better)
    """
    # This is a placeholder implementation
    print("Note: This is a placeholder FID implementation.")
    print("For accurate FID calculation, use a pre-trained Inception v3 model.")
    
    # Return dummy value for now
    return 50.0


def evaluate_model_quality(model: nn.Module, test_loader, device: torch.device,
                          num_samples: int = 100) -> Dict[str, Any]:
    """
    Comprehensive model quality evaluation
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        num_samples: Number of samples to generate for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Calculate test NLL and BPD
    total_nll = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            logits = model(images)
            nll = calculate_nll(logits, images)
            
            total_nll += nll * batch_size
            total_samples += batch_size
    
    avg_nll = total_nll / total_samples
    avg_bpd = calculate_bits_per_dimension(avg_nll, 32, 32, 3)
    
    # Generate samples for quality metrics
    samples = sample_from_model(model, (3, 32, 32), device, num_samples=num_samples)
    
    # Calculate quality metrics (placeholders for now)
    is_mean, is_std = calculate_inception_score(samples)
    
    # Get some real samples for FID calculation
    real_samples = []
    for images, _ in test_loader:
        real_samples.append(images)
        if len(real_samples) * images.size(0) >= num_samples:
            break
    
    real_samples = torch.cat(real_samples, dim=0)[:num_samples]
    fid_score = calculate_fid_score(real_samples, samples.cpu())
    
    return {
        'test_nll': avg_nll,
        'test_bpd': avg_bpd,
        'inception_score_mean': is_mean,
        'inception_score_std': is_std,
        'fid_score': fid_score,
        'num_samples_evaluated': num_samples
    }


def save_experiment_config(config: Dict[str, Any], filepath: str):
    """Save experiment configuration to JSON file"""
    import json
    with open(filepath, 'w') as f:
        # Convert any non-serializable objects to strings
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_config[key] = value
            else:
                serializable_config[key] = str(value)
        
        json.dump(serializable_config, f, indent=2)


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test NLL calculation
    batch_size, channels, num_classes, height, width = 2, 3, 256, 4, 4
    logits = torch.randn(batch_size, channels, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, channels, height, width))
    
    nll = calculate_nll(logits, targets)
    bpd = calculate_bits_per_dimension(nll, height, width, channels)
    
    print(f"Test NLL: {nll:.4f}")
    print(f"Test BPD: {bpd:.4f}")
    
    # Test loss function
    criterion = PixelCrossEntropyLoss()
    loss = criterion(logits, targets)
    print(f"Test Loss: {loss.item():.4f}")
    
    print("All utility functions working correctly!")