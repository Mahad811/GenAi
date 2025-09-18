import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from typing import List, Tuple, Dict
import re


class ShakespeareDataset(Dataset):
    def __init__(self, split: str = 'train', seq_len: int = 100, vocab_size: int = None):
        """
        Shakespeare dataset for next-word prediction
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            seq_len: Sequence length for training
            vocab_size: Maximum vocabulary size (None for full vocab)
        """
        self.seq_len = seq_len
        
        # Load dataset - use direct download approach
        try:
            # Try loading with ignore_verifications
            self.dataset = load_dataset('karpathy/tiny_shakespeare', ignore_verifications=True)
            if split == 'train':
                self.text = self.dataset['train']['text']
            elif split == 'validation':
                self.text = self.dataset['validation']['text']
            else:  # test
                self.text = self.dataset['test']['text']
        except Exception as e:
            # Fallback: use a simple Shakespeare text sample
            print(f"Warning: {e}")
            print("Using fallback Shakespeare text...")
            self.text = """
            To be, or not to be, that is the question:
            Whether 'tis nobler in the mind to suffer
            The slings and arrows of outrageous fortune,
            Or to take arms against a sea of troubles
            And by opposing end them. To die—to sleep,
            No more; and by a sleep to say we end
            The heart-ache and the thousand natural shocks
            That flesh is heir to: 'tis a consummation
            Devoutly to be wish'd. To die, to sleep;
            To sleep, perchance to dream—ay, there's the rub:
            For in that sleep of death what dreams may come,
            When we have shuffled off this mortal coil,
            Must give us pause—there's the respect
            That makes calamity of so long life.
            """
        
        # Create character-level vocabulary
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        
        # Limit vocabulary size if specified
        if vocab_size is not None and vocab_size < self.vocab_size:
            # Keep most frequent characters
            char_counts = {}
            for char in self.text:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Sort by frequency and take top vocab_size
            sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
            self.chars = [char for char, _ in sorted_chars[:vocab_size]]
            self.vocab_size = len(self.chars)
        
        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Convert text to indices
        self.text_indices = [self.char_to_idx.get(char, 0) for char in self.text]
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.text_indices) - seq_len):
            seq = self.text_indices[i:i + seq_len]
            target = self.text_indices[i + seq_len]
            self.sequences.append(seq)
            self.targets.append(target)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sequence, target
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary information"""
        return {
            'vocab_size': self.vocab_size,
            'chars': self.chars,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char
        }
    
    def decode_sequence(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])
    
    def encode_sequence(self, text: str) -> List[int]:
        """Convert text to indices"""
        return [self.char_to_idx.get(char, 0) for char in text]


def create_data_loaders(train_split: str = 'train', val_split: str = 'validation', 
                       seq_len: int = 100, batch_size: int = 32, 
                       vocab_size: int = None, num_workers: int = 2) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation data loaders
    
    Args:
        train_split: Training split name
        val_split: Validation split name  
        seq_len: Sequence length
        batch_size: Batch size
        vocab_size: Vocabulary size limit
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, vocab_info)
    """
    # Create datasets
    train_dataset = ShakespeareDataset(split=train_split, seq_len=seq_len, vocab_size=vocab_size)
    val_dataset = ShakespeareDataset(split=val_split, seq_len=seq_len, vocab_size=vocab_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get vocabulary info from train dataset
    vocab_info = train_dataset.get_vocab_info()
    
    return train_loader, val_loader, vocab_info


def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing
    
    Args:
        text: Raw text
        
    Returns:
        Preprocessed text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    
    return text.strip()


if __name__ == '__main__':
    # Test the dataset
    print("Testing Shakespeare Dataset...")
    
    # Create a small test dataset
    train_loader, val_loader, vocab_info = create_data_loaders(
        seq_len=50, 
        batch_size=4, 
        vocab_size=100
    )
    
    print(f"Vocabulary size: {vocab_info['vocab_size']}")
    print(f"Sample characters: {vocab_info['chars'][:20]}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch_idx, (sequences, targets) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Target shape: {targets.shape}")
        
        # Decode first sequence
        first_seq = sequences[0].tolist()
        decoded = train_loader.dataset.decode_sequence(first_seq)
        print(f"First sequence: '{decoded[:50]}...'")
        print(f"Target: '{vocab_info['idx_to_char'][targets[0].item()]}'")
        
        if batch_idx >= 2:  # Only show first few batches
            break