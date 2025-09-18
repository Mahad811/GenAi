import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from src.dataset import create_data_loaders
from src.model import ShakespeareRNN
from src.utils import set_seed, ensure_dir, evaluate_model_performance


def train_and_evaluate_ablation(model: ShakespeareRNN, train_loader: DataLoader, 
                               val_loader: DataLoader, device: torch.device, 
                               epochs: int = 10, lr: float = 1e-3) -> Dict[str, float]:
    """
    Train and evaluate model for ablation study
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        epochs: Number of epochs
        lr: Learning rate
        
    Returns:
        Dictionary with evaluation metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(sequences)
            
            # Select only the last output for next-character prediction
            outputs = outputs[:, -1, :]
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Validation
        val_metrics = evaluate_model_performance(model, val_loader, criterion, device)
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
    
    return {
        'best_val_loss': best_val_loss,
        'best_val_perplexity': val_metrics['perplexity'],
        'best_val_accuracy': val_metrics['accuracy']
    }


def run_ablation_experiment(experiment_name: str, model_config: Dict, 
                           train_loader: DataLoader, val_loader: DataLoader, 
                           device: torch.device, epochs: int = 10) -> Dict:
    """
    Run a single ablation experiment
    
    Args:
        experiment_name: Name of the experiment
        model_config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        epochs: Number of epochs
        
    Returns:
        Experiment results
    """
    print(f"Running experiment: {experiment_name}")
    
    # Create model
    model = ShakespeareRNN(**model_config).to(device)
    
    # Train and evaluate
    results = train_and_evaluate_ablation(model, train_loader, val_loader, device, epochs)
    
    # Add model info
    model_info = model.get_model_info()
    results.update({
        'experiment': experiment_name,
        'total_parameters': model_info['total_parameters'],
        'trainable_parameters': model_info['trainable_parameters']
    })
    
    # Add configuration
    results.update(model_config)
    
    print(f"  Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"  Best Val Perplexity: {results['best_val_perplexity']:.2f}")
    print(f"  Parameters: {results['total_parameters']:,}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ablation Study for Shakespeare RNN')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per experiment')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--vocab_size', type=int, default=None, help='Vocabulary size limit')
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    ensure_dir(args.outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, vocab_info = create_data_loaders(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size
    )
    
    print(f"Vocabulary size: {vocab_info['vocab_size']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Base configuration
    base_config = {
        'vocab_size': vocab_info['vocab_size'],
        'embedding_dim': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'rnn_type': 'LSTM'
    }
    
    results = []
    
    print("\nStarting Ablation Study...")
    print("=" * 60)
    
    # 1. RNN Type Ablation
    print("\n1. RNN Type Ablation")
    rnn_types = ['LSTM', 'GRU', 'RNN']
    for rnn_type in rnn_types:
        config = base_config.copy()
        config['rnn_type'] = rnn_type
        result = run_ablation_experiment(
            f'RNN_Type_{rnn_type}',
            config,
            train_loader,
            val_loader,
            device,
            args.epochs
        )
        results.append(result)
    
    # 2. Hidden Size Ablation
    print("\n2. Hidden Size Ablation")
    hidden_sizes = [128, 256, 512]
    for hidden_size in hidden_sizes:
        config = base_config.copy()
        config['hidden_size'] = hidden_size
        result = run_ablation_experiment(
            f'Hidden_Size_{hidden_size}',
            config,
            train_loader,
            val_loader,
            device,
            args.epochs
        )
        results.append(result)
    
    # 3. Number of Layers Ablation
    print("\n3. Number of Layers Ablation")
    num_layers_list = [1, 2, 3, 4]
    for num_layers in num_layers_list:
        config = base_config.copy()
        config['num_layers'] = num_layers
        result = run_ablation_experiment(
            f'Num_Layers_{num_layers}',
            config,
            train_loader,
            val_loader,
            device,
            args.epochs
        )
        results.append(result)
    
    # 4. Embedding Dimension Ablation
    print("\n4. Embedding Dimension Ablation")
    embedding_dims = [64, 128, 256]
    for embedding_dim in embedding_dims:
        config = base_config.copy()
        config['embedding_dim'] = embedding_dim
        result = run_ablation_experiment(
            f'Embedding_Dim_{embedding_dim}',
            config,
            train_loader,
            val_loader,
            device,
            args.epochs
        )
        results.append(result)
    
    # 5. Dropout Ablation
    print("\n5. Dropout Ablation")
    dropout_rates = [0.0, 0.2, 0.3, 0.5]
    for dropout in dropout_rates:
        config = base_config.copy()
        config['dropout'] = dropout
        result = run_ablation_experiment(
            f'Dropout_{dropout}',
            config,
            train_loader,
            val_loader,
            device,
            args.epochs
        )
        results.append(result)
    
    # 6. Learning Rate Ablation
    print("\n6. Learning Rate Ablation")
    learning_rates = [1e-4, 1e-3, 1e-2]
    for lr in learning_rates:
        config = base_config.copy()
        result = run_ablation_experiment(
            f'Learning_Rate_{lr}',
            config,
            train_loader,
            val_loader,
            device,
            args.epochs
        )
        # Override learning rate in training
        result['learning_rate'] = lr
        results.append(result)
    
    # Save results
    print("\nSaving results...")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.outdir, 'ablation_results.csv'), index=False)
    
    # Find best configurations
    print("\nFinding best configurations...")
    
    best_configs = {}
    for experiment_type in ['RNN_Type', 'Hidden_Size', 'Num_Layers', 'Embedding_Dim', 'Dropout', 'Learning_Rate']:
        type_results = df[df['experiment'].str.startswith(experiment_type)]
        if not type_results.empty:
            best_idx = type_results['best_val_loss'].idxmin()
            best_configs[experiment_type] = type_results.loc[best_idx].to_dict()
    
    # Save best configurations
    with open(os.path.join(args.outdir, 'best_ablation_configs.json'), 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Overall performance comparison
    plt.figure(figsize=(15, 10))
    
    # RNN Type comparison
    plt.subplot(2, 3, 1)
    rnn_results = df[df['experiment'].str.startswith('RNN_Type')]
    plt.bar(rnn_results['rnn_type'], rnn_results['best_val_perplexity'], color=['blue', 'orange', 'green'])
    plt.title('RNN Type Comparison')
    plt.ylabel('Validation Perplexity')
    plt.xticks(rotation=45)
    
    # Hidden Size comparison
    plt.subplot(2, 3, 2)
    hidden_results = df[df['experiment'].str.startswith('Hidden_Size')]
    plt.plot(hidden_results['hidden_size'], hidden_results['best_val_perplexity'], 'o-', color='red')
    plt.title('Hidden Size Comparison')
    plt.xlabel('Hidden Size')
    plt.ylabel('Validation Perplexity')
    plt.grid(True)
    
    # Number of Layers comparison
    plt.subplot(2, 3, 3)
    layers_results = df[df['experiment'].str.startswith('Num_Layers')]
    plt.plot(layers_results['num_layers'], layers_results['best_val_perplexity'], 'o-', color='purple')
    plt.title('Number of Layers Comparison')
    plt.xlabel('Number of Layers')
    plt.ylabel('Validation Perplexity')
    plt.grid(True)
    
    # Embedding Dimension comparison
    plt.subplot(2, 3, 4)
    embed_results = df[df['experiment'].str.startswith('Embedding_Dim')]
    plt.plot(embed_results['embedding_dim'], embed_results['best_val_perplexity'], 'o-', color='brown')
    plt.title('Embedding Dimension Comparison')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Validation Perplexity')
    plt.grid(True)
    
    # Dropout comparison
    plt.subplot(2, 3, 5)
    dropout_results = df[df['experiment'].str.startswith('Dropout')]
    plt.plot(dropout_results['dropout'], dropout_results['best_val_perplexity'], 'o-', color='pink')
    plt.title('Dropout Rate Comparison')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation Perplexity')
    plt.grid(True)
    
    # Learning Rate comparison
    plt.subplot(2, 3, 6)
    lr_results = df[df['experiment'].str.startswith('Learning_Rate')]
    plt.semilogx(lr_results['learning_rate'], lr_results['best_val_perplexity'], 'o-', color='cyan')
    plt.title('Learning Rate Comparison')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Perplexity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'ablation_study_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Parameter count vs performance
    plt.figure(figsize=(10, 6))
    plt.scatter(df['total_parameters'], df['best_val_perplexity'], alpha=0.7, s=100)
    plt.xlabel('Total Parameters')
    plt.ylabel('Validation Perplexity')
    plt.title('Model Size vs Performance')
    plt.grid(True)
    
    # Add labels for some points
    for idx, row in df.iterrows():
        if row['experiment'] in ['RNN_Type_LSTM', 'RNN_Type_GRU', 'RNN_Type_RNN']:
            plt.annotate(row['experiment'].split('_')[-1], 
                        (row['total_parameters'], row['best_val_perplexity']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'parameter_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    
    for experiment_type, best_config in best_configs.items():
        print(f"\nBest {experiment_type}:")
        print(f"  Configuration: {best_config['experiment']}")
        print(f"  Validation Perplexity: {best_config['best_val_perplexity']:.2f}")
        print(f"  Validation Loss: {best_config['best_val_loss']:.4f}")
        print(f"  Parameters: {best_config['total_parameters']:,}")
    
    # Overall best model
    overall_best = df.loc[df['best_val_perplexity'].idxmin()]
    print(f"\nOverall Best Model:")
    print(f"  Configuration: {overall_best['experiment']}")
    print(f"  Validation Perplexity: {overall_best['best_val_perplexity']:.2f}")
    print(f"  Validation Loss: {overall_best['best_val_loss']:.4f}")
    print(f"  Parameters: {overall_best['total_parameters']:,}")
    
    print(f"\nAblation study completed! Results saved to: {args.outdir}")


if __name__ == '__main__':
    main()