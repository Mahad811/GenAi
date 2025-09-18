# Q2: Shakespeare RNN for Next-Word Prediction

This project implements a Recurrent Neural Network (RNN) for next-word prediction on the Shakespeare text dataset from Hugging Face. The implementation includes character-level tokenization, custom word embeddings, and comprehensive evaluation.

## Project Structure

```
Q2/
├── src/
│   ├── dataset.py          # Shakespeare dataset preprocessing
│   ├── model.py            # RNN/LSTM model implementation
│   ├── train.py            # Basic training script
│   ├── train_optimal.py    # Training with optimal hyperparameters
│   ├── evaluate.py         # Model evaluation and text generation
│   ├── ablation_study.py   # Hyperparameter ablation study
│   └── utils.py            # Utility functions
├── outputs/                # Generated outputs and results
├── colab_runner.ipynb      # Google Colab execution notebook
└── README.md              # This file
```

## Features

### 1. Dataset Processing

- **Character-level tokenization** of Shakespeare text
- **Custom vocabulary creation** with configurable size limits
- **Sequence generation** for next-word prediction
- **Train/validation/test splits** with proper data loaders

### 2. Model Architecture

- **Configurable RNN types**: LSTM, GRU, or vanilla RNN
- **Custom embedding layer** (no pre-trained embeddings)
- **Multi-layer support** with dropout regularization
- **Flexible architecture** with configurable dimensions

### 3. Training Pipeline

- **Comprehensive training loop** with validation monitoring
- **Learning rate scheduling** with ReduceLROnPlateau
- **Gradient clipping** for training stability
- **Model checkpointing** and best model saving
- **Training curve visualization**

### 4. Evaluation Metrics

- **Perplexity calculation** (primary metric)
- **Accuracy measurement** for next-character prediction
- **N-gram perplexity** for comparison
- **Text generation quality analysis**

### 5. Ablation Study

- **RNN type comparison** (LSTM vs GRU vs RNN)
- **Architecture hyperparameter tuning**:
  - Hidden size (128, 256, 512)
  - Number of layers (1, 2, 3, 4)
  - Embedding dimension (64, 128, 256)
  - Dropout rate (0.0, 0.2, 0.3, 0.5)
  - Learning rate (1e-4, 1e-3, 1e-2)

### 6. Text Generation

- **Temperature-controlled sampling** for creativity control
- **Top-k and top-p filtering** for better quality
- **Multiple seed text support** for diverse generation
- **Interactive generation** in Colab notebook

## Usage

### Local Execution (Tasks 1-5)

1. **Basic Training**:

```bash
python -m src.train --epochs 20 --batch_size 32 --seq_len 100
```

2. **Ablation Study**:

```bash
python -m src.ablation_study --outdir outputs --epochs 10
```

3. **Optimal Training**:

```bash
python -m src.train_optimal --epochs 20 --best_config_file outputs/best_ablation_configs.json
```

4. **Model Evaluation**:

```bash
python -m src.evaluate --model_path outputs/optimal_model.pt --num_samples 10
```

### Google Colab Execution (Task 6)

1. Open `colab_runner.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Choose setup option (Drive, GitHub, or ZIP upload)
4. Run all cells sequentially

## Key Implementation Details

### Dataset Processing

- Uses HuggingFace `karpathy/tiny_shakespeare` dataset
- Character-level tokenization (not word-level)
- Configurable sequence length and vocabulary size
- Proper train/validation splits

### Model Architecture

```python
ShakespeareRNN(
    vocab_size=vocab_size,      # Vocabulary size
    embedding_dim=128,          # Embedding dimension
    hidden_size=256,            # Hidden state size
    num_layers=2,               # Number of RNN layers
    dropout=0.3,                # Dropout rate
    rnn_type='LSTM'             # RNN type
)
```

### Training Configuration

- **Optimizer**: Adam with configurable learning rate
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau
- **Gradient Clipping**: max_norm=1.0
- **Batch Size**: Configurable (default 32)
- **Sequence Length**: Configurable (default 100)

### Evaluation Metrics

1. **Perplexity**: Primary metric for language modeling
2. **Accuracy**: Character-level prediction accuracy
3. **N-gram Perplexity**: For comparison with traditional methods
4. **Text Quality**: Vocabulary coverage, repetition analysis

## Expected Outputs

### Files Generated

- `best_model.pt` - Best model checkpoint
- `optimal_model.pt` - Optimal hyperparameter model
- `training_curves.png` - Training/validation curves
- `ablation_study_results.png` - Ablation study visualizations
- `evaluation_results.png` - Evaluation metrics and analysis
- `generated_texts.txt` - Sample generated texts
- `model_comparison.csv` - Performance comparison data

### Key Metrics

- **Validation Perplexity**: Target < 10.0 for good performance
- **Text Generation**: Coherent, contextually appropriate text
- **Model Size**: Balanced between performance and efficiency

## Requirements

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Datasets
- NumPy, Pandas, Matplotlib
- Scikit-learn

## Notes

- The model uses **character-level** tokenization, not word-level
- **Custom embeddings** are trained from scratch (no pre-trained)
- **Sequence-to-sequence** architecture for next-character prediction
- **Comprehensive evaluation** including both quantitative and qualitative metrics
- **Ablation study** provides insights into optimal hyperparameters

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or sequence length
2. **Slow training**: Use GPU acceleration or reduce model size
3. **Poor text quality**: Increase training epochs or adjust temperature
4. **Import errors**: Ensure all dependencies are installed

This implementation provides a complete pipeline for training and evaluating RNNs for next-word prediction, following the same structure and quality standards as Q1.
