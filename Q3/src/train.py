import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Tuple, Dict

from src.dataset import create_cifar10_loaders
from src.pixelcnn import PixelCNN
from src.row_lstm import RowLSTM
from src.diagonal_bilstm import DiagonalBiLSTM
from src.utils import set_seed, ensure_dir, calculate_bits_per_dimension, calculate_nll


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   optimizer: optim.Optimizer, device: torch.device, 
                   epoch: int, print_freq: int = 100) -> Tuple[float, float]:
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_samples = 0
    
    start_time = time.time()
    
    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)  # (B, C, H, W) with values [0, 255]
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)  # (B, C, num_classes, H, W)
        
        # Calculate loss
        # Reshape for CrossEntropyLoss: (B*C*H*W, num_classes) and (B*C*H*W,)
        B, C, num_classes, H, W = logits.shape
        logits_flat = logits.permute(0, 1, 3, 4, 2).contiguous().view(-1, num_classes)
        targets_flat = images.view(-1).long()
        
        loss = criterion(logits_flat, targets_flat)
        nll = calculate_nll(logits, images)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate statistics
        total_loss += loss.item() * batch_size
        total_nll += nll * batch_size
        total_samples += batch_size
        
        # Print progress
        if batch_idx % print_freq == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, '
                  f'Loss: {loss.item():.4f}, NLL: {nll:.4f}, '
                  f'Time: {elapsed:.1f}s')
    
    avg_loss = total_loss / total_samples
    avg_nll = total_nll / total_samples
    
    return avg_loss, avg_nll


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_nll = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Forward pass
            logits = model(images)
            
            # Calculate metrics
            # Reshape for CrossEntropyLoss: (B*C*H*W, num_classes) and (B*C*H*W,)
            B, C, num_classes, H, W = logits.shape
            logits_flat = logits.permute(0, 1, 3, 4, 2).contiguous().view(-1, num_classes)
            targets_flat = images.view(-1).long()
            
            loss = criterion(logits_flat, targets_flat)
            nll = calculate_nll(logits, images)
            
            total_loss += loss.item() * batch_size
            total_nll += nll * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    avg_nll = total_nll / total_samples
    avg_bpd = calculate_bits_per_dimension(avg_nll, 32, 32, 3)  # CIFAR-10 dimensions
    
    return avg_loss, avg_nll, avg_bpd


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Create model based on type"""
    if model_type.lower() == 'pixelcnn':
        return PixelCNN(**kwargs)
    elif model_type.lower() == 'row_lstm':
        return RowLSTM(**kwargs)
    elif model_type.lower() == 'diagonal_bilstm':
        return DiagonalBiLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train PixelRNN/CNN Models')
    parser.add_argument('--model_type', type=str, default='pixelcnn', 
                       choices=['pixelcnn', 'row_lstm', 'diagonal_bilstm'],
                       help='Model type to train')
    parser.add_argument('--data_path', type=str, 
                       default='/home/mahad/Desktop/genai_A1/Q3/cifar-10-python.tar.gz',
                       help='Path to CIFAR-10 dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--print_freq', type=int, default=100, help='Print frequency')
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    ensure_dir(args.outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Training model: {args.model_type}')
    
    # Create data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = create_cifar10_loaders(
        args.data_path, batch_size=args.batch_size
    )
    
    # Create model
    model_kwargs = {
        'input_channels': 3,
        'hidden_channels': args.hidden_channels,
        'num_layers': args.num_layers,
        'num_classes': 256
    }
    
    model = create_model(args.model_type, **model_kwargs).to(device)
    
    print(f"\nModel created:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    train_nlls = []
    val_losses = []
    val_nlls = []
    val_bpds = []
    
    best_val_nll = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_nll = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.print_freq
        )
        
        # Validate
        val_loss, val_nll, val_bpd = validate(model, test_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_nll)
        
        # Store metrics
        train_losses.append(train_loss)
        train_nlls.append(train_nll)
        val_losses.append(val_loss)
        val_nlls.append(val_nll)
        val_bpds.append(val_bpd)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f'Epoch {epoch:2d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f}, NLL: {train_nll:.4f} | '
              f'Val Loss: {val_loss:.4f}, NLL: {val_nll:.4f}, BPD: {val_bpd:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
              f'Time: {epoch_time:.1f}s')
        
        # Save best model
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nll': val_nll,
                'val_bpd': val_bpd,
                'model_info': model_info,
                'args': args
            }, os.path.join(args.outdir, f'best_{args.model_type}.pt'))
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nll': val_nll,
                'val_bpd': val_bpd,
            }, os.path.join(args.outdir, f'{args.model_type}_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_nll': val_nll,
        'val_bpd': val_bpd,
        'model_info': model_info,
        'args': args
    }, os.path.join(args.outdir, f'final_{args.model_type}.pt'))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.model_type} - Training Loss')
    plt.legend()
    plt.grid(True)
    
    # NLL curves
    plt.subplot(1, 3, 2)
    plt.plot(range(1, args.epochs + 1), train_nlls, label='Train NLL', color='blue')
    plt.plot(range(1, args.epochs + 1), val_nlls, label='Val NLL', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.title(f'{args.model_type} - NLL')
    plt.legend()
    plt.grid(True)
    
    # Bits per dimension
    plt.subplot(1, 3, 3)
    plt.plot(range(1, args.epochs + 1), val_bpds, label='Val BPD', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Bits per Dimension')
    plt.title(f'{args.model_type} - Bits/Dimension')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f'{args.model_type}_training_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training history
    import pandas as pd
    history = pd.DataFrame({
        'epoch': range(1, args.epochs + 1),
        'train_loss': train_losses,
        'train_nll': train_nlls,
        'val_loss': val_losses,
        'val_nll': val_nlls,
        'val_bpd': val_bpds
    })
    history.to_csv(os.path.join(args.outdir, f'{args.model_type}_training_history.csv'), index=False)
    
    print(f"\nTraining completed!")
    print(f"Best validation NLL: {best_val_nll:.4f}")
    print(f"Best validation BPD: {calculate_bits_per_dimension(best_val_nll, 32, 32, 3):.4f}")
    print(f"Results saved to: {args.outdir}")


if __name__ == '__main__':
    main()