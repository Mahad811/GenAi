import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from src.dataset import CIFAR10HFDataset
from src.model import SimpleCNN
from src.utils import set_seed, ensure_dir


def train_and_evaluate(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    set_seed(42)
    ensure_dir(args.outdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_ds = CIFAR10HFDataset(split='train[:80%]')
    val_ds = CIFAR10HFDataset(split='train[80%:90%]')

    # Hyperparameter configurations to test
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    base_filters = [16, 32, 64]
    num_layers = [3, 5, 7]

    results = []

    print("Starting Ablation Study...")
    print("=" * 50)

    # Test learning rates
    print("Testing Learning Rates...")
    for lr in learning_rates:
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
        
        model = SimpleCNN(num_classes=10, num_layers=3, base_filters=32).to(device)
        acc = train_and_evaluate(model, train_loader, val_loader, device, epochs=args.epochs, lr=lr)
        
        results.append({
            'experiment': 'learning_rate',
            'parameter': 'lr',
            'value': lr,
            'accuracy': acc
        })
        print(f"LR {lr}: {acc:.4f}")

    # Test batch sizes
    print("Testing Batch Sizes...")
    for bs in batch_sizes:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)
        
        model = SimpleCNN(num_classes=10, num_layers=3, base_filters=32).to(device)
        acc = train_and_evaluate(model, train_loader, val_loader, device, epochs=args.epochs)
        
        results.append({
            'experiment': 'batch_size',
            'parameter': 'batch_size',
            'value': bs,
            'accuracy': acc
        })
        print(f"Batch Size {bs}: {acc:.4f}")

    # Test base filters
    print("Testing Number of Filters...")
    for bf in base_filters:
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
        
        model = SimpleCNN(num_classes=10, num_layers=3, base_filters=bf).to(device)
        acc = train_and_evaluate(model, train_loader, val_loader, device, epochs=args.epochs)
        
        results.append({
            'experiment': 'base_filters',
            'parameter': 'base_filters',
            'value': bf,
            'accuracy': acc
        })
        print(f"Base Filters {bf}: {acc:.4f}")

    # Test number of layers
    print("Testing Number of Layers...")
    for nl in num_layers:
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
        
        model = SimpleCNN(num_classes=10, num_layers=nl, base_filters=32).to(device)
        acc = train_and_evaluate(model, train_loader, val_loader, device, epochs=args.epochs)
        
        results.append({
            'experiment': 'num_layers',
            'parameter': 'num_layers',
            'value': nl,
            'accuracy': acc
        })
        print(f"Num Layers {nl}: {acc:.4f}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.outdir, 'ablation_results.csv'), index=False)

    # Find best hyperparameters
    best_lr = df[df['experiment'] == 'learning_rate'].loc[df[df['experiment'] == 'learning_rate']['accuracy'].idxmax(), 'value']
    best_bs = df[df['experiment'] == 'batch_size'].loc[df[df['experiment'] == 'batch_size']['accuracy'].idxmax(), 'value']
    best_bf = df[df['experiment'] == 'base_filters'].loc[df[df['experiment'] == 'base_filters']['accuracy'].idxmax(), 'value']
    best_nl = df[df['experiment'] == 'num_layers'].loc[df[df['experiment'] == 'num_layers']['accuracy'].idxmax(), 'value']

    best_params = {
        'learning_rate': float(best_lr),
        'batch_size': int(best_bs),
        'base_filters': int(best_bf),
        'num_layers': int(best_nl)
    }

    with open(os.path.join(args.outdir, 'best_hyperparameters.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    print("\n" + "=" * 50)
    print("Ablation Study Complete!")
    print("Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")


if __name__ == '__main__':
    main()
