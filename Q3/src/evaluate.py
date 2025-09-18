import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List

from src.dataset import create_cifar10_loaders
from src.pixelcnn import PixelCNN
from src.row_lstm import RowLSTM
from src.diagonal_bilstm import DiagonalBiLSTM
from src.utils import (set_seed, ensure_dir, calculate_nll, calculate_bits_per_dimension,
                      sample_from_model, visualize_samples, evaluate_model_quality,
                      compare_models, PixelCrossEntropyLoss)


def load_model(model_path: str, device: torch.device) -> tuple:
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        (model, model_info, args): Loaded model, info, and training args
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model info and args
    model_info = checkpoint['model_info']
    args = checkpoint.get('args', None)
    
    # Create model based on type
    model_type = model_info['model_type'].lower().replace(' ', '_')
    
    if 'pixelcnn' in model_type:
        model = PixelCNN(
            input_channels=model_info['input_channels'],
            hidden_channels=model_info['hidden_channels'],
            num_layers=model_info['num_layers'],
            num_classes=model_info['num_classes']
        )
    elif 'row' in model_type and 'lstm' in model_type:
        model = RowLSTM(
            input_channels=model_info['input_channels'],
            hidden_channels=model_info['hidden_channels'],
            num_layers=model_info['num_layers'],
            num_classes=model_info['num_classes']
        )
    elif 'diagonal' in model_type and 'bilstm' in model_type:
        model = DiagonalBiLSTM(
            input_channels=model_info['input_channels'],
            hidden_channels=model_info['hidden_channels'],
            num_layers=model_info['num_layers'],
            num_classes=model_info['num_classes']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, model_info, args


def evaluate_single_model(model: nn.Module, test_loader: DataLoader, 
                         device: torch.device, model_name: str) -> Dict:
    """Evaluate a single model"""
    print(f"\nEvaluating {model_name}...")
    
    model.eval()
    criterion = PixelCrossEntropyLoss()
    
    total_loss = 0.0
    total_nll = 0.0
    total_samples = 0
    
    # Calculate test metrics
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Forward pass
            logits = model(images)
            
            # Calculate metrics
            loss = criterion(logits, images)
            nll = calculate_nll(logits, images)
            
            total_loss += loss.item() * batch_size
            total_nll += nll * batch_size
            total_samples += batch_size
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)}, "
                      f"Loss: {loss.item():.4f}, NLL: {nll:.4f}")
    
    avg_loss = total_loss / total_samples
    avg_nll = total_nll / total_samples
    avg_bpd = calculate_bits_per_dimension(avg_nll, 32, 32, 3)
    
    print(f"  Final Results:")
    print(f"    Test Loss: {avg_loss:.4f}")
    print(f"    Test NLL: {avg_nll:.4f}")
    print(f"    Test BPD: {avg_bpd:.4f}")
    
    return {
        'model_name': model_name,
        'test_loss': avg_loss,
        'test_nll': avg_nll,
        'test_bpd': avg_bpd,
        'parameters': sum(p.numel() for p in model.parameters())
    }


def generate_samples_comparison(models: Dict[str, nn.Module], device: torch.device,
                               output_dir: str, num_samples: int = 16):
    """Generate and compare samples from different models"""
    print(f"\nGenerating {num_samples} samples from each model...")
    
    for model_name, model in models.items():
        print(f"  Generating samples from {model_name}...")
        
        # Generate samples
        samples = sample_from_model(model, (3, 32, 32), device, 
                                  temperature=1.0, num_samples=num_samples)
        
        # Visualize samples
        save_path = os.path.join(output_dir, f'{model_name}_samples.png')
        visualize_samples(samples, save_path=save_path, 
                         title=f'{model_name} Generated Samples', nrow=4)
        
        print(f"    Samples saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PixelRNN/CNN Models')
    parser.add_argument('--data_path', type=str,
                       default='/home/mahad/Desktop/genai_A1/Q3/cifar-10-python.tar.gz',
                       help='Path to CIFAR-10 dataset')
    parser.add_argument('--model_dir', type=str, default='outputs',
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=16, 
                       help='Number of samples to generate')
    parser.add_argument('--models', nargs='+', 
                       default=['pixelcnn', 'row_lstm', 'diagonal_bilstm'],
                       help='Models to evaluate')
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    ensure_dir(args.output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load test data
    print("Loading CIFAR-10 test dataset...")
    _, test_loader = create_cifar10_loaders(args.data_path, batch_size=args.batch_size)
    
    # Load models
    models = {}
    results = {}
    
    for model_name in args.models:
        model_path = os.path.join(args.model_dir, f'best_{model_name}.pt')
        
        if os.path.exists(model_path):
            print(f"\nLoading {model_name} from {model_path}")
            model, model_info, train_args = load_model(model_path, device)
            models[model_name] = model
            
            print(f"Model info:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            
            # Evaluate model
            result = evaluate_single_model(model, test_loader, device, model_name)
            results[model_name] = result
            
        else:
            print(f"Warning: Model file not found: {model_path}")
    
    if not models:
        print("No models found to evaluate!")
        return
    
    # Save evaluation results
    results_df = pd.DataFrame(list(results.values()))
    results_csv_path = os.path.join(args.output_dir, 'model_comparison.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Test Loss':<12} {'Test NLL':<12} {'Test BPD':<12} {'Parameters':<12}")
    print(f"{'-'*80}")
    
    for model_name, result in results.items():
        params_m = result['parameters'] / 1e6  # Convert to millions
        print(f"{model_name:<20} {result['test_loss']:<12.4f} "
              f"{result['test_nll']:<12.4f} {result['test_bpd']:<12.4f} {params_m:<12.2f}M")
    
    # Create comparison plots
    comparison_plot_path = os.path.join(args.output_dir, 'model_comparison.png')
    
    # Prepare data for plotting
    plot_results = {}
    for model_name, result in results.items():
        plot_results[model_name] = {
            'val_nll': result['test_nll'],
            'val_bpd': result['test_bpd'],
            'parameters': result['parameters']
        }
    
    compare_models(plot_results, save_path=comparison_plot_path)
    print(f"Comparison plot saved to: {comparison_plot_path}")
    
    # Generate sample comparisons
    generate_samples_comparison(models, device, args.output_dir, args.num_samples)
    
    # Create comprehensive evaluation report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("PixelRNN/CNN Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Dataset: CIFAR-10\n")
        f.write(f"Test samples: {len(test_loader.dataset)}\n")
        f.write(f"Evaluation device: {device}\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 30 + "\n")
        
        # Sort by BPD (lower is better)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_bpd'])
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            f.write(f"{i}. {model_name}:\n")
            f.write(f"   Test NLL: {result['test_nll']:.4f}\n")
            f.write(f"   Test BPD: {result['test_bpd']:.4f}\n")
            f.write(f"   Parameters: {result['parameters']:,}\n")
            f.write(f"   Efficiency (BPD/M params): {result['test_bpd']/(result['parameters']/1e6):.4f}\n\n")
        
        # Best model
        best_model = sorted_results[0]
        f.write(f"Best Model: {best_model[0]} (BPD: {best_model[1]['test_bpd']:.4f})\n\n")
        
        f.write("Key Findings:\n")
        f.write("-" * 15 + "\n")
        f.write("- Lower Bits per Dimension (BPD) indicates better generative modeling performance\n")
        f.write("- Negative Log-Likelihood (NLL) measures how well the model fits the data\n")
        f.write("- Parameter efficiency shows model complexity vs performance trade-off\n")
    
    print(f"Evaluation report saved to: {report_path}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED!")
    print(f"Results saved in: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()