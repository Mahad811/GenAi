import os
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import CIFAR10HFDataset
from src.model import SimpleCNN
from src.utils import set_seed, ensure_dir


def visualize_feature_maps(model, sample_input, device, outdir):
    model.eval()
    with torch.no_grad():
        sample_input = sample_input.unsqueeze(0).to(device)
        feature_maps = model.get_feature_maps(sample_input)
    
    # Plot original image
    original = sample_input.squeeze().cpu()
    # Denormalize for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    original = original * std + mean
    original = torch.clamp(original, 0, 1)
    
    plt.figure(figsize=(15, 10))
    
    # Show original image
    plt.subplot(2, 4, 1)
    plt.imshow(original.permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    
    # Show feature maps from different layers
    subplot_idx = 2  # Start from position 2
    for i, fmap in enumerate(feature_maps[:3]):  # Show first 3 conv layers
        # Take first few channels of each feature map
        fmap_subset = fmap[0]  # [channels, H, W]
        
        # Show 2-3 filters per layer to fit in remaining 7 subplot positions
        num_filters_to_show = min(2, fmap_subset.shape[0]) if i < 2 else min(3, fmap_subset.shape[0])
        
        for j in range(num_filters_to_show):
            if subplot_idx <= 8:  # Ensure we don't exceed subplot limit
                plt.subplot(2, 4, subplot_idx)
                plt.imshow(fmap_subset[j], cmap='viridis')
                plt.title(f'Layer {i+1}, Filter {j+1}')
                plt.axis('off')
                subplot_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'feature_maps.png'))
    plt.close()
    
    # Create a summary plot showing average activation per layer
    plt.figure(figsize=(12, 6))
    layer_activations = []
    for i, fmap in enumerate(feature_maps):
        avg_activation = fmap.mean().item()
        layer_activations.append(avg_activation)
        plt.subplot(1, len(feature_maps), i+1)
        plt.imshow(fmap[0].mean(dim=0), cmap='viridis')
        plt.title(f'Layer {i+1}\nAvg: {avg_activation:.3f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'layer_activations.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='outputs/best_model.pt')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--base_filters', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (not used but accepted for compatibility)')
    parser.add_argument('--num_samples', type=int, default=6, help='Number of samples to visualize')
    parser.add_argument('--outdir', type=str, default='outputs')
    args = parser.parse_args()

    set_seed(42)
    ensure_dir(args.outdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a sample from test set
    test_ds = CIFAR10HFDataset(split='test')
    sample_input, sample_label = test_ds[0]  # Get first test sample

    # Load model
    model = SimpleCNN(num_classes=10, num_layers=args.num_layers, base_filters=args.base_filters)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Visualize feature maps
    visualize_feature_maps(model, sample_input, device, args.outdir)
    
    print("Feature maps saved to outputs/feature_maps.png and outputs/layer_activations.png")
    print("\nAnalysis of Convolutional Layers:")
    print("- Layer 1: Detects low-level features like edges, corners, and simple patterns")
    print("- Layer 2: Combines edges to form more complex shapes and textures") 
    print("- Layer 3: Recognizes higher-level object parts and patterns specific to classes")
    print("- Each layer builds upon previous layers to create increasingly abstract representations")


if __name__ == '__main__':
    main()
