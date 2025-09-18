import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from src.dataset import create_data_loaders
from src.model import ShakespeareRNN
from src.utils import set_seed, ensure_dir, evaluate_model_performance, generate_text


def calculate_ngram_perplexity(model: torch.nn.Module, data_loader: DataLoader, 
                              vocab_info: Dict, n: int = 3, device: torch.device = None) -> float:
    """
    Calculate n-gram perplexity for comparison
    
    Args:
        model: Trained model
        data_loader: Data loader
        vocab_info: Vocabulary information
        n: N-gram size
        device: Device to run on
        
    Returns:
        N-gram perplexity
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    ngram_counts = {}
    total_ngrams = 0
    
    with torch.no_grad():
        for sequences, _ in data_loader:
            sequences = sequences.to(device)
            
            # Convert to text
            for seq in sequences:
                text = vocab_info['idx_to_char'][seq[0].item()]
                for i in range(1, len(seq)):
                    text += vocab_info['idx_to_char'][seq[i].item()]
                
                # Count n-grams
                for i in range(len(text) - n + 1):
                    ngram = text[i:i+n]
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
                    total_ngrams += 1
    
    # Calculate perplexity
    log_prob_sum = 0.0
    for ngram, count in ngram_counts.items():
        prob = count / total_ngrams
        log_prob_sum += count * np.log(prob)
    
    perplexity = np.exp(-log_prob_sum / total_ngrams)
    return perplexity


def analyze_generated_text(generated_texts: List[str], vocab_info: Dict) -> Dict:
    """
    Analyze generated text for quality metrics
    
    Args:
        generated_texts: List of generated text samples
        vocab_info: Vocabulary information
        
    Returns:
        Dictionary with analysis metrics
    """
    all_text = ' '.join(generated_texts)
    
    # Basic statistics
    total_chars = len(all_text)
    unique_chars = len(set(all_text))
    vocab_coverage = unique_chars / len(vocab_info['chars'])
    
    # Word/character statistics
    words = all_text.split()
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    avg_sentence_length = np.mean([len(sentence.split()) for sentence in all_text.split('.') if sentence.strip()])
    
    # Repetition analysis
    char_repetition = {}
    for char in all_text:
        char_repetition[char] = char_repetition.get(char, 0) + 1
    
    most_common_char = max(char_repetition.items(), key=lambda x: x[1]) if char_repetition else ('', 0)
    repetition_ratio = most_common_char[1] / total_chars if total_chars > 0 else 0
    
    return {
        'total_characters': total_chars,
        'unique_characters': unique_chars,
        'vocabulary_coverage': vocab_coverage,
        'average_word_length': avg_word_length,
        'average_sentence_length': avg_sentence_length,
        'most_common_character': most_common_char[0],
        'character_repetition_ratio': repetition_ratio,
        'number_of_samples': len(generated_texts)
    }


def generate_multiple_samples(model: torch.nn.Module, vocab_info: Dict, 
                            seed_texts: List[str], max_length: int = 100, 
                            temperature: float = 1.0, device: torch.device = None) -> List[str]:
    """
    Generate multiple text samples for evaluation
    
    Args:
        model: Trained model
        vocab_info: Vocabulary information
        seed_texts: List of seed texts
        max_length: Maximum generation length
        temperature: Sampling temperature
        device: Device to run on
        
    Returns:
        List of generated texts
    """
    generated_texts = []
    
    for seed in seed_texts:
        generated = generate_text(
            model=model,
            vocab_info=vocab_info,
            seed_text=seed,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        generated_texts.append(generated)
    
    return generated_texts


def main():
    parser = argparse.ArgumentParser(description='Evaluate Shakespeare RNN')
    parser.add_argument('--model_path', type=str, default='outputs/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--outdir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of text samples to generate')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    ensure_dir(args.outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model checkpoint
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Create model
    model_info = checkpoint['model_info']
    model = ShakespeareRNN(
        vocab_size=model_info['vocab_size'],
        embedding_dim=model_info['embedding_dim'],
        hidden_size=model_info['hidden_size'],
        num_layers=model_info['num_layers'],
        dropout=0.0,  # No dropout during evaluation
        rnn_type=model_info['rnn_type']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab_info = checkpoint['vocab_info']
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Create data loaders for evaluation
    print("Creating data loaders...")
    train_loader, val_loader, _ = create_data_loaders(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=None  # Use full vocabulary
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    
    # Test set evaluation (using validation split as test)
    test_metrics = evaluate_model_performance(model, val_loader, criterion, device)
    
    print(f"\nTest Set Performance:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Perplexity: {test_metrics['perplexity']:.2f}")
    
    # Calculate n-gram perplexity for comparison
    print("Calculating n-gram perplexity...")
    bigram_perp = calculate_ngram_perplexity(model, val_loader, vocab_info, n=2, device=device)
    trigram_perp = calculate_ngram_perplexity(model, val_loader, vocab_info, n=3, device=device)
    
    print(f"  Bigram Perplexity: {bigram_perp:.2f}")
    print(f"  Trigram Perplexity: {trigram_perp:.2f}")
    
    # Generate text samples
    print(f"\nGenerating {args.num_samples} text samples...")
    
    # Different seed texts for variety
    seed_texts = [
        "To be or not to",
        "Once upon a time",
        "The quick brown fox",
        "In the beginning",
        "All the world's a",
        "To infinity and",
        "The best of times",
        "It was the best",
        "Call me Ishmael",
        "It is a truth"
    ]
    
    # Generate samples
    generated_texts = generate_multiple_samples(
        model=model,
        vocab_info=vocab_info,
        seed_texts=seed_texts[:args.num_samples],
        max_length=args.max_length,
        temperature=args.temperature,
        device=device
    )
    
    # Analyze generated text
    analysis = analyze_generated_text(generated_texts, vocab_info)
    
    print(f"\nGenerated Text Analysis:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    print("\nSaving results...")
    
    # Save evaluation metrics
    metrics_data = {
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_perplexity': test_metrics['perplexity'],
        'bigram_perplexity': bigram_perp,
        'trigram_perplexity': trigram_perp,
        **analysis
    }
    
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(os.path.join(args.outdir, 'evaluation_metrics.csv'), index=False)
    
    # Save generated texts
    with open(os.path.join(args.outdir, 'generated_texts.txt'), 'w') as f:
        f.write("Generated Text Samples\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (seed, generated) in enumerate(zip(seed_texts[:args.num_samples], generated_texts)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Seed: '{seed}'\n")
            f.write(f"Generated: '{generated}'\n")
            f.write("-" * 30 + "\n\n")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Model performance comparison
    plt.subplot(2, 2, 1)
    metrics = ['Test Perplexity', 'Bigram Perplexity', 'Trigram Perplexity']
    values = [test_metrics['perplexity'], bigram_perp, trigram_perp]
    bars = plt.bar(metrics, values, color=['blue', 'orange', 'green'])
    plt.title('Perplexity Comparison')
    plt.ylabel('Perplexity')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom')
    
    # Text analysis
    plt.subplot(2, 2, 2)
    analysis_metrics = ['Vocabulary Coverage', 'Avg Word Length', 'Avg Sentence Length']
    analysis_values = [analysis['vocabulary_coverage'], 
                      analysis['average_word_length'], 
                      analysis['average_sentence_length']]
    plt.bar(analysis_metrics, analysis_values, color=['purple', 'red', 'brown'])
    plt.title('Generated Text Analysis')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Character distribution
    plt.subplot(2, 2, 3)
    all_text = ' '.join(generated_texts)
    char_counts = {}
    for char in all_text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Get top 10 most common characters
    top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    chars, counts = zip(*top_chars)
    
    plt.bar(range(len(chars)), counts, color='skyblue')
    plt.title('Top 10 Most Common Characters')
    plt.xlabel('Characters')
    plt.ylabel('Count')
    plt.xticks(range(len(chars)), chars)
    
    # Model architecture info
    plt.subplot(2, 2, 4)
    arch_info = [
        f"Vocab Size: {model_info['vocab_size']}",
        f"Embedding Dim: {model_info['embedding_dim']}",
        f"Hidden Size: {model_info['hidden_size']}",
        f"Layers: {model_info['num_layers']}",
        f"RNN Type: {model_info['rnn_type']}",
        f"Parameters: {model_info['total_parameters']:,}"
    ]
    
    plt.text(0.1, 0.9, '\n'.join(arch_info), transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    plt.title('Model Architecture')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print sample generated texts
    print(f"\nSample Generated Texts:")
    print("=" * 50)
    for i, (seed, generated) in enumerate(zip(seed_texts[:5], generated_texts[:5])):
        print(f"\nSample {i+1}:")
        print(f"Seed: '{seed}'")
        print(f"Generated: '{generated}'")
    
    print(f"\nEvaluation completed! Results saved to: {args.outdir}")


if __name__ == '__main__':
    main()