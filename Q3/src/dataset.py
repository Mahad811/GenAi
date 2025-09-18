import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pickle
import os
from typing import Tuple, Dict, Optional
import tarfile


class CIFAR10PixelDataset(Dataset):
    """
    CIFAR-10 dataset for pixel-level autoregressive modeling
    """
    def __init__(self, data_path: str, split: str = 'train', transform=None):
        """
        Initialize CIFAR-10 dataset for PixelRNN/CNN
        
        Args:
            data_path: Path to cifar-10-python.tar.gz
            split: 'train' or 'test'
            transform: Optional transforms
        """
        self.split = split
        self.transform = transform
        
        # Extract and load CIFAR-10 data
        self.data, self.labels = self._load_cifar10(data_path, split)
        
        # Convert to proper format for pixel modeling
        self.data = self._preprocess_data(self.data)
    
    def _load_cifar10(self, data_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CIFAR-10 data from tar.gz file"""
        
        # Extract tar.gz if needed
        extract_dir = os.path.dirname(data_path)
        cifar_dir = os.path.join(extract_dir, 'cifar-10-batches-py')
        
        if not os.path.exists(cifar_dir):
            print(f"Extracting {data_path}...")
            with tarfile.open(data_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        if split == 'train':
            # Load training batches
            data_list = []
            labels_list = []
            
            for i in range(1, 6):  # data_batch_1 to data_batch_5
                batch_file = os.path.join(cifar_dir, f'data_batch_{i}')
                batch = unpickle(batch_file)
                data_list.append(batch[b'data'])
                labels_list.append(batch[b'labels'])
            
            data = np.concatenate(data_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            
        else:  # test
            test_batch = unpickle(os.path.join(cifar_dir, 'test_batch'))
            data = test_batch[b'data']
            labels = np.array(test_batch[b'labels'])
        
        return data, labels
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess CIFAR-10 data for pixel modeling
        
        Args:
            data: Raw CIFAR-10 data (N, 3072)
            
        Returns:
            Preprocessed data (N, 3, 32, 32) with values in [0, 255]
        """
        # Reshape from (N, 3072) to (N, 3, 32, 32)
        data = data.reshape(-1, 3, 32, 32)
        
        # Keep pixel values as integers [0, 255] for discrete softmax
        return data.astype(np.uint8)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            image: (3, 32, 32) tensor with values [0, 255]
            label: class label (not used in generative modeling)
        """
        image = self.data[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        image = torch.from_numpy(image).long()  # Long tensor for discrete values
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_cifar10_loaders(data_path: str, batch_size: int = 32, 
                          num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 data loaders for PixelRNN training
    
    Args:
        data_path: Path to cifar-10-python.tar.gz
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        train_loader, test_loader
    """
    
    # Create datasets
    train_dataset = CIFAR10PixelDataset(data_path, split='train')
    test_dataset = CIFAR10PixelDataset(data_path, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Image shape: {train_dataset[0][0].shape}")
    print(f"  Pixel value range: [0, 255] (discrete)")
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    data_path = '/home/mahad/Desktop/genai_A1/Q3/cifar-10-python.tar.gz'
    
    if os.path.exists(data_path):
        train_loader, test_loader = create_cifar10_loaders(data_path, batch_size=4)
        
        # Test a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Image dtype: {images.dtype}")
            print(f"Pixel values range: [{images.min()}, {images.max()}]")
            
            if batch_idx >= 2:
                break
    else:
        print(f"Dataset not found at {data_path}")
        print("Please ensure the cifar-10-python.tar.gz file is in the Q3 directory")