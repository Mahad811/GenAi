import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

from src.dataset import create_data_loaders
from src.model import ShakespeareRNN
from src.utils import set_seed, ensure_dir, calculate_perplexity


def train_one_epoch(model: ShakespeareRNN, loader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    for batch_idx, (sequences, targets) in enumerate(loader):
        sequences, targets = sequences.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs, _ = model(sequences)
        
        # For next-character prediction, use only the last output
        # outputs: [batch_size, seq_len, vocab_size]
        # targets: [batch_size] - single target per sequence
        outputs = outputs[:, -1, :]  # Take last output: [batch_size, vocab_size]
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * targets.size(0)
        total_tokens += targets.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


def validate(model: ShakespeareRNN, loader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for sequences, targets in loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Forward pass
            outputs, _ = model(sequences)
            
            # For next-character prediction, use only the last output
            outputs = outputs[:, -1, :]  # Take last output: [batch_size, vocab_size]
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)
    
    avg_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description='Train Shakespeare RNN')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='RNN type')
    parser.add_argument('--vocab_size', type=int, default=None, help='Vocabulary size limit')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    ensure_dir(args.outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, vocab_info = create_data_loaders(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size
    )
    
    print(f"Vocabulary size: {vocab_info['vocab_size']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = ShakespeareRNN(
        vocab_size=vocab_info['vocab_size'],
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type
    ).to(device)
    
    print(f"\nModel created:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_perp = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_perp = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_perplexities.append(train_perp)
        val_perplexities.append(val_perp)
        
        # Print progress
        print(f'Epoch {epoch:2d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} (Perp: {train_perp:.2f}) | '
              f'Val Loss: {val_loss:.4f} (Perp: {val_perp:.2f}) | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab_info': vocab_info,
                'model_info': model_info,
                'args': args
            }, os.path.join(args.outdir, 'best_model.pt'))
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.outdir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'vocab_info': vocab_info,
        'model_info': model_info,
        'args': args
    }, os.path.join(args.outdir, 'final_model.pt'))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Perplexity curves
    plt.subplot(1, 3, 2)
    plt.plot(range(1, args.epochs + 1), train_perplexities, label='Train Perplexity', color='blue')
    plt.plot(range(1, args.epochs + 1), val_perplexities, label='Val Perplexity', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True)
    
    # Learning rate
    plt.subplot(1, 3, 3)
    lrs = [optimizer.param_groups[0]['lr']] * args.epochs  # Simplified for plotting
    plt.plot(range(1, args.epochs + 1), lrs, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training history
    import pandas as pd
    history = pd.DataFrame({
        'epoch': range(1, args.epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_perplexity': train_perplexities,
        'val_perplexity': val_perplexities
    })
    history.to_csv(os.path.join(args.outdir, 'training_history.csv'), index=False)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {calculate_perplexity(best_val_loss):.2f}")
    print(f"Results saved to: {args.outdir}")


if __name__ == '__main__':
    main()