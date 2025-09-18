import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

from src.dataset import create_data_loaders
from src.model import ShakespeareRNN
from src.utils import set_seed, ensure_dir, evaluate_model_performance, generate_text


def train_one_epoch(model: ShakespeareRNN, loader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer, device: torch.device) -> tuple:
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
    return avg_loss


def validate(model: ShakespeareRNN, loader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> tuple:
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
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Shakespeare RNN with Optimal Hyperparameters')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--best_config_file', type=str, default='outputs/best_ablation_configs.json', 
                       help='Path to best ablation configurations')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--vocab_size', type=int, default=None, help='Vocabulary size limit')
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    ensure_dir(args.outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load best configurations
    print("Loading best ablation configurations...")
    with open(args.best_config_file, 'r') as f:
        best_configs = json.load(f)
    
    # Create optimal configuration by combining best settings
    optimal_config = {
        'vocab_size': None,  # Will be set from dataset
        'embedding_dim': best_configs['Embedding_Dim']['embedding_dim'],
        'hidden_size': best_configs['Hidden_Size']['hidden_size'],
        'num_layers': best_configs['Num_Layers']['num_layers'],
        'dropout': best_configs['Dropout']['dropout'],
        'rnn_type': best_configs['RNN_Type']['rnn_type']
    }
    
    optimal_lr = best_configs['Learning_Rate']['learning_rate']
    
    print("Optimal Configuration:")
    for key, value in optimal_config.items():
        if key != 'vocab_size':
            print(f"  {key}: {value}")
    print(f"  learning_rate: {optimal_lr}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, vocab_info = create_data_loaders(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size
    )
    
    optimal_config['vocab_size'] = vocab_info['vocab_size']
    
    print(f"Vocabulary size: {vocab_info['vocab_size']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create optimal model
    print("\nCreating optimal model...")
    optimal_model = ShakespeareRNN(**optimal_config).to(device)
    
    model_info = optimal_model.get_model_info()
    print(f"Model created with {model_info['total_parameters']:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(optimal_model.parameters(), lr=optimal_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    optimal_train_losses = []
    optimal_val_losses = []
    optimal_val_perplexities = []
    
    best_val_loss = float('inf')
    
    print(f"\nStarting optimal training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(optimal_model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(optimal_model, val_loader, criterion, device)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store metrics
        optimal_train_losses.append(train_loss)
        optimal_val_losses.append(val_loss)
        optimal_val_perplexities.append(val_perplexity)
        
        # Print progress
        print(f'Epoch {epoch:2d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} (Perp: {val_perplexity:.2f}) | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': optimal_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab_info': vocab_info,
                'model_info': model_info,
                'optimal_config': optimal_config,
                'learning_rate': optimal_lr
            }, os.path.join(args.outdir, 'optimal_model.pt'))
    
    # Train baseline model for comparison
    print("\nTraining baseline model for comparison...")
    baseline_config = {
        'vocab_size': vocab_info['vocab_size'],
        'embedding_dim': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'rnn_type': 'LSTM'
    }
    
    baseline_model = ShakespeareRNN(**baseline_config).to(device)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3)
    
    baseline_val_losses = []
    baseline_val_perplexities = []
    
    for epoch in range(1, args.epochs + 1):
        # Train baseline
        train_loss = train_one_epoch(baseline_model, train_loader, criterion, baseline_optimizer, device)
        
        # Validate baseline
        val_loss = validate(baseline_model, val_loader, criterion, device)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        baseline_val_losses.append(val_loss)
        baseline_val_perplexities.append(val_perplexity)
        
        if epoch % 5 == 0:
            print(f'Baseline Epoch {epoch}/{args.epochs} | Val Loss: {val_loss:.4f} (Perp: {val_perplexity:.2f})')
    
    # Generate sample texts for comparison
    print("\nGenerating sample texts...")
    
    seed_texts = [
        "To be or not to",
        "Once upon a time",
        "The quick brown fox"
    ]
    
    # Generate with optimal model
    optimal_samples = []
    for seed in seed_texts:
        generated = generate_text(
            model=optimal_model,
            vocab_info=vocab_info,
            seed_text=seed,
            max_length=50,
            temperature=0.8,
            device=device
        )
        optimal_samples.append(generated)
    
    # Generate with baseline model
    baseline_samples = []
    for seed in seed_texts:
        generated = generate_text(
            model=baseline_model,
            vocab_info=vocab_info,
            seed_text=seed,
            max_length=50,
            temperature=0.8,
            device=device
        )
        baseline_samples.append(generated)
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Training curves
    plt.subplot(2, 3, 1)
    plt.plot(range(1, args.epochs + 1), optimal_train_losses, label='Optimal Train Loss', color='blue')
    plt.plot(range(1, args.epochs + 1), optimal_val_losses, label='Optimal Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimal Model Training')
    plt.legend()
    plt.grid(True)
    
    # Model comparison
    plt.subplot(2, 3, 2)
    plt.plot(range(1, args.epochs + 1), optimal_val_perplexities, label='Optimal Model', color='blue')
    plt.plot(range(1, args.epochs + 1), baseline_val_perplexities, label='Baseline Model', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Perplexity')
    plt.title('Model Comparison')
    plt.legend()
    plt.grid(True)
    
    # Final performance comparison
    plt.subplot(2, 3, 3)
    models = ['Baseline', 'Optimal']
    final_perplexities = [baseline_val_perplexities[-1], optimal_val_perplexities[-1]]
    bars = plt.bar(models, final_perplexities, color=['red', 'blue'])
    plt.title('Final Validation Perplexity')
    plt.ylabel('Perplexity')
    
    # Add value labels
    for bar, value in zip(bars, final_perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom')
    
    # Model architecture comparison
    plt.subplot(2, 3, 4)
    arch_comparison = {
        'Embedding Dim': [baseline_config['embedding_dim'], optimal_config['embedding_dim']],
        'Hidden Size': [baseline_config['hidden_size'], optimal_config['hidden_size']],
        'Num Layers': [baseline_config['num_layers'], optimal_config['num_layers']],
        'Dropout': [baseline_config['dropout'], optimal_config['dropout']]
    }
    
    x = range(len(arch_comparison))
    width = 0.35
    
    for i, (param, values) in enumerate(arch_comparison.items()):
        plt.bar(i - width/2, values[0], width, label='Baseline' if i == 0 else '', color='red', alpha=0.7)
        plt.bar(i + width/2, values[1], width, label='Optimal' if i == 0 else '', color='blue', alpha=0.7)
    
    plt.xlabel('Parameters')
    plt.ylabel('Value')
    plt.title('Architecture Comparison')
    plt.xticks(x, arch_comparison.keys(), rotation=45)
    plt.legend()
    
    # Parameter count comparison
    plt.subplot(2, 3, 5)
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    optimal_params = sum(p.numel() for p in optimal_model.parameters())
    
    plt.bar(['Baseline', 'Optimal'], [baseline_params, optimal_params], color=['red', 'blue'])
    plt.title('Model Size Comparison')
    plt.ylabel('Parameters')
    
    # Add value labels
    for i, (model, params) in enumerate(zip(['Baseline', 'Optimal'], [baseline_params, optimal_params])):
        plt.text(i, params + max(baseline_params, optimal_params) * 0.01, 
                f'{params:,}', ha='center', va='bottom')
    
    # Text generation comparison
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.9, 'Text Generation Comparison:', transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold')
    
    y_pos = 0.8
    for i, (seed, optimal, baseline) in enumerate(zip(seed_texts, optimal_samples, baseline_samples)):
        plt.text(0.1, y_pos, f'Seed: "{seed}"', transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, y_pos - 0.05, f'Optimal: "{optimal[:30]}..."', transform=plt.gca().transAxes, 
                fontsize=9, color='blue')
        plt.text(0.1, y_pos - 0.1, f'Baseline: "{baseline[:30]}..."', transform=plt.gca().transAxes, 
                fontsize=9, color='red')
        y_pos -= 0.2
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'optimal_vs_baseline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison results
    final_optimal_perp = optimal_val_perplexities[-1]
    final_baseline_perp = baseline_val_perplexities[-1]
    improvement = final_baseline_perp - final_optimal_perp  # Lower perplexity is better
    improvement_pct = (improvement / final_baseline_perp) * 100
    
    comparison_data = {
        'Model': ['Baseline', 'Optimal'],
        'Final_Validation_Perplexity': [final_baseline_perp, final_optimal_perp],
        'Final_Validation_Loss': [baseline_val_losses[-1], optimal_val_losses[-1]],
        'Total_Parameters': [baseline_params, optimal_params],
        'Improvement': [0.0, improvement],
        'Improvement_Percentage': [0.0, improvement_pct]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(args.outdir, 'model_comparison.csv'), index=False)
    
    # Save generated texts
    with open(os.path.join(args.outdir, 'text_generation_comparison.txt'), 'w') as f:
        f.write("Text Generation Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (seed, optimal, baseline) in enumerate(zip(seed_texts, optimal_samples, baseline_samples)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Seed: '{seed}'\n")
            f.write(f"Optimal Model: '{optimal}'\n")
            f.write(f"Baseline Model: '{baseline}'\n")
            f.write("-" * 30 + "\n\n")
    
    print(f"\nFinal Results:")
    print(f"Baseline Model Perplexity: {final_baseline_perp:.2f}")
    print(f"Optimal Model Perplexity: {final_optimal_perp:.2f}")
    print(f"Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
    print(f"Baseline Parameters: {baseline_params:,}")
    print(f"Optimal Parameters: {optimal_params:,}")
    
    print(f"\nOptimal training completed! Results saved to: {args.outdir}")


if __name__ == '__main__':
    main()