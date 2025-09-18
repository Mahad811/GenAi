#!/usr/bin/env python3
"""
Test script to verify Q2 implementation works correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from src.dataset import create_data_loaders
from src.model import ShakespeareRNN
from src.utils import set_seed, calculate_perplexity, generate_text


def test_dataset():
    """Test dataset loading and processing"""
    print("Testing dataset...")
    
    try:
        train_loader, val_loader, vocab_info = create_data_loaders(
            seq_len=50,
            batch_size=4,
            vocab_size=100
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Vocabulary size: {vocab_info['vocab_size']}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Test a batch
        for sequences, targets in train_loader:
            print(f"  Batch shapes: sequences {sequences.shape}, targets {targets.shape}")
            break
            
        return True, vocab_info
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False, None


def test_model(vocab_info):
    """Test model creation and forward pass"""
    print("\nTesting model...")
    
    try:
        model = ShakespeareRNN(
            vocab_size=vocab_info['vocab_size'],
            embedding_dim=64,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            rnn_type='LSTM'
        )
        
        print(f"✓ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size, seq_len = 4, 50
        x = torch.randint(0, vocab_info['vocab_size'], (batch_size, seq_len))
        
        output, hidden = model(x)
        print(f"  Forward pass: input {x.shape} -> output {output.shape}")
        
        return True, model
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False, None


def test_text_generation(model, vocab_info):
    """Test text generation"""
    print("\nTesting text generation...")
    
    try:
        seed_text = "To be or not to"
        generated = generate_text(
            model=model,
            vocab_info=vocab_info,
            seed_text=seed_text,
            max_length=20,
            temperature=0.8,
            device=torch.device('cpu')
        )
        
        print(f"✓ Text generation successful")
        print(f"  Seed: '{seed_text}'")
        print(f"  Generated: '{generated}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Text generation test failed: {e}")
        return False


def test_utils():
    """Test utility functions"""
    print("\nTesting utilities...")
    
    try:
        # Test perplexity calculation
        loss = 2.0
        perp = calculate_perplexity(loss)
        print(f"✓ Perplexity calculation: loss {loss} -> perplexity {perp:.2f}")
        
        # Test seed setting
        set_seed(42)
        print(f"✓ Seed setting successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Q2 Implementation Test Suite")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Test dataset
    dataset_ok, vocab_info = test_dataset()
    if not dataset_ok:
        print("\n❌ Dataset test failed. Stopping.")
        return False
    
    # Test model
    model_ok, model = test_model(vocab_info)
    if not model_ok:
        print("\n❌ Model test failed. Stopping.")
        return False
    
    # Test text generation
    generation_ok = test_text_generation(model, vocab_info)
    
    # Test utilities
    utils_ok = test_utils()
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"  Dataset: {'✓' if dataset_ok else '✗'}")
    print(f"  Model: {'✓' if model_ok else '✗'}")
    print(f"  Text Generation: {'✓' if generation_ok else '✗'}")
    print(f"  Utils: {'✓' if utils_ok else '✗'}")
    
    all_passed = all([dataset_ok, model_ok, generation_ok, utils_ok])
    
    if all_passed:
        print("\n🎉 All tests passed! Implementation is working correctly.")
        print("\nYou can now run:")
        print("  python -m src.train --epochs 5  # Quick training test")
        print("  python -m src.ablation_study --epochs 3  # Quick ablation test")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)