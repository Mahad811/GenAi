import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from src.dataset import CIFAR10HFDataset
from src.model import SimpleCNN
from src.utils import set_seed, ensure_dir


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--outdir', type=str, default='outputs')
    parser.add_argument('--best_params_file', type=str, default='outputs/best_hyperparameters.json')
    args = parser.parse_args()

    set_seed(42)
    ensure_dir(args.outdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best hyperparameters
    with open(args.best_params_file, 'r') as f:
        best_params = json.load(f)

    print("Training with optimal hyperparameters:")
    print(f"  Learning Rate: {best_params['learning_rate']}")
    print(f"  Batch Size: {best_params['batch_size']}")
    print(f"  Base Filters: {best_params['base_filters']}")
    print(f"  Num Layers: {best_params['num_layers']}")

    # Load datasets with optimal batch size
    train_ds = CIFAR10HFDataset(split='train[:90%]')
    val_ds = CIFAR10HFDataset(split='train[90%:]')
    train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=best_params['batch_size'], shuffle=False, num_workers=2)

    # Create optimal model
    optimal_model = SimpleCNN(
        num_classes=10,
        num_layers=best_params['num_layers'],
        base_filters=best_params['base_filters']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(optimal_model.parameters(), lr=best_params['learning_rate'])

    # Train optimal model
    optimal_train_losses = []
    optimal_val_losses = []
    optimal_val_accs = []

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(optimal_model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(optimal_model, val_loader, criterion, device)
        optimal_train_losses.append(tr_loss)
        optimal_val_losses.append(va_loss)
        optimal_val_accs.append(va_acc)
        
        print(f'Epoch {epoch}/{args.epochs} - train_loss: {tr_loss:.4f} acc: {tr_acc:.4f} | val_loss: {va_loss:.4f} acc: {va_acc:.4f}')
        
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(optimal_model.state_dict(), os.path.join(args.outdir, 'optimal_model.pt'))

    # Train baseline model for comparison (default hyperparameters)
    print("\nTraining baseline model for comparison...")
    baseline_model = SimpleCNN(num_classes=10, num_layers=3, base_filters=32).to(device)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3)
    baseline_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    baseline_val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    baseline_val_accs = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(baseline_model, baseline_loader, criterion, baseline_optimizer, device)
        va_loss, va_acc = validate(baseline_model, baseline_val_loader, criterion, device)
        baseline_val_accs.append(va_acc)

    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), optimal_train_losses, label='Optimal Train Loss')
    plt.plot(range(1, args.epochs + 1), optimal_val_losses, label='Optimal Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimal Model Training')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), optimal_val_accs, label='Optimal Model')
    plt.plot(range(1, args.epochs + 1), baseline_val_accs, label='Baseline Model')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'optimal_vs_baseline.png'))
    
    # Save comparison results
    final_optimal_acc = max(optimal_val_accs)
    final_baseline_acc = max(baseline_val_accs)
    improvement = final_optimal_acc - final_baseline_acc
    
    comparison_data = {
        'Model': ['Baseline', 'Optimal'],
        'Best_Validation_Accuracy': [final_baseline_acc, final_optimal_acc],
        'Improvement': [0.0, improvement]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(args.outdir, 'model_comparison.csv'), index=False)
    
    print(f"\nFinal Results:")
    print(f"Baseline Model Accuracy: {final_baseline_acc:.4f}")
    print(f"Optimal Model Accuracy: {final_optimal_acc:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")


if __name__ == '__main__':
    main()
