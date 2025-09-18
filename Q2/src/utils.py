import os
import random
import numpy as np
import torch
import math
from typing import List, Dict, Any
import json


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(loss)


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy"""
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, 
                      top_k: int = None, top_p: float = None) -> int:
    """
    Sample from logits with optional temperature, top-k, and top-p filtering
    
    Args:
        logits: Model output logits
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens
        top_p: Keep tokens with cumulative probability <= top_p
        
    Returns:
        Sampled token index
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k filtering
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, top_k_indices, top_k_logits)
    
    # Apply top-p (nucleus) filtering
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    # Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_text(model: torch.nn.Module, vocab_info: Dict, seed_text: str, 
                 max_length: int = 100, temperature: float = 1.0, 
                 top_k: int = None, top_p: float = None, device: torch.device = None) -> str:
    """
    Generate text using the model
    
    Args:
        model: Trained model
        vocab_info: Vocabulary information
        seed_text: Starting text
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        device: Device to run on
        
    Returns:
        Generated text
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    char_to_idx = vocab_info['char_to_idx']
    idx_to_char = vocab_info['idx_to_char']
    
    generated = seed_text
    hidden = None
    
    # Convert seed to indices
    seed_indices = [char_to_idx.get(char, 0) for char in seed_text]
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_seq = torch.tensor([seed_indices[-1]], dtype=torch.long).unsqueeze(0).to(device)
            
            # Forward pass
            if hasattr(model, 'generate'):
                # Use model's generate method if available
                output, hidden = model(input_seq, hidden)
            else:
                # Fallback for other model types
                output = model(input_seq)
                hidden = None
            
            # Get logits for last character
            if len(output.shape) == 3:  # [batch, seq, vocab]
                logits = output[0, -1, :]
            else:  # [batch, vocab]
                logits = output[0, :]
            
            # Sample next character
            next_idx = sample_from_logits(logits, temperature, top_k, top_p)
            
            # Add to generated text
            next_char = idx_to_char.get(next_idx, '')
            generated += next_char
            seed_indices.append(next_idx)
            
            # Stop if we hit a natural stopping point
            if next_char in ['.', '!', '?'] and len(generated) > len(seed_text) + 10:
                break
    
    return generated


def evaluate_model_performance(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                              criterion: torch.nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model performance on a dataset
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Forward pass
            if hasattr(model, 'forward') and 'hidden' in model.forward.__code__.co_varnames:
                outputs, _ = model(sequences)
            else:
                outputs = model(sequences)
            
            # For next-character prediction, use only the last output
            outputs = outputs[:, -1, :]  # Take last output: [batch_size, vocab_size]
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=-1)
            correct_predictions += (predictions == targets).sum().item()
    
    avg_loss = total_loss / total_tokens
    accuracy = correct_predictions / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }


def save_experiment_config(config: Dict[str, Any], filepath: str):
    """Save experiment configuration to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_experiment_config(filepath: str) -> Dict[str, Any]:
    """Load experiment configuration from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model: torch.nn.Module, input_shape: tuple = None):
    """Print model summary"""
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    # Parameter count
    param_counts = count_parameters(model)
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    
    # Model info
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"\nModel Configuration:")
        for key, value in model_info.items():
            if key not in ['total_parameters', 'trainable_parameters']:
                print(f"  {key}: {value}")
    
    print("=" * 60)


if __name__ == '__main__':
    # Test utility functions
    print("Testing utility functions...")
    
    # Test perplexity calculation
    test_loss = 2.0
    perp = calculate_perplexity(test_loss)
    print(f"Loss: {test_loss}, Perplexity: {perp:.2f}")
    
    # Test sampling
    logits = torch.randn(10)
    sampled_idx = sample_from_logits(logits, temperature=1.0)
    print(f"Sampled index: {sampled_idx}")
    
    # Test time formatting
    print(f"Time formatting: {format_time(3661.5)}")
    
    print("Utility functions test completed!")