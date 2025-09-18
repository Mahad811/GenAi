import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from src.dataset import CIFAR10HFDataset
from src.model import SimpleCNN
from src.utils import set_seed, ensure_dir


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='outputs/best_model.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--base_filters', type=int, default=32)
    parser.add_argument('--outdir', type=str, default='outputs')
    args = parser.parse_args()

    set_seed(42)
    ensure_dir(args.outdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset
    test_ds = CIFAR10HFDataset(split='test')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load model
    model = SimpleCNN(num_classes=10, num_layers=args.num_layers, base_filters=args.base_filters)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Get predictions
    preds, labels = evaluate_model(model, test_loader, device)

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'confusion_matrix.png'))
    plt.close()

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    # Create metrics table
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    print("Model Performance:")
    print(metrics_df.to_string(index=False))
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(args.outdir, 'metrics.csv'), index=False)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None)
    
    per_class_data = {
        'Class': class_names,
        'Precision': precision_per_class,
        'Recall': recall_per_class,
        'F1-Score': f1_per_class
    }
    per_class_df = pd.DataFrame(per_class_data)
    
    print("\nPer-class Performance:")
    print(per_class_df.to_string(index=False))
    
    per_class_df.to_csv(os.path.join(args.outdir, 'per_class_metrics.csv'), index=False)


if __name__ == '__main__':
    main()
