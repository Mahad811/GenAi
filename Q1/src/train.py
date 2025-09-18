import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--base_filters', type=int, default=32)
    parser.add_argument('--outdir', type=str, default='outputs')
    args = parser.parse_args()

    set_seed(42)
    ensure_dir(args.outdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = CIFAR10HFDataset(split='train[:90%]')
    val_ds = CIFAR10HFDataset(split='train[90%:]')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=10, num_layers=args.num_layers, base_filters=args.base_filters).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        print(f'Epoch {epoch}/{args.epochs} - train_loss: {tr_loss:.4f} acc: {tr_acc:.4f} | val_loss: {va_loss:.4f} acc: {va_acc:.4f}')
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best_model.pt'))

    # Plot training/validation loss
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='train_loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'loss_curve.png'))


if __name__ == '__main__':
    main()