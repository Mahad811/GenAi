# Q2 Command Reference

This file contains all the commands needed to run Q2 tasks locally and on Google Colab.

## Local Execution (Tasks 1-5)

### Prerequisites

```bash
# Install dependencies
pip install -r ../requirements.txt

# Test implementation
python test_implementation.py
```

### Task 1: Load and Preprocess Dataset

```bash
# Test dataset loading
python -c "from src.dataset import create_data_loaders; train_loader, val_loader, vocab_info = create_data_loaders(seq_len=50, batch_size=4); print(f'Vocab size: {vocab_info[\"vocab_size\"]}, Train batches: {len(train_loader)}')"
```

### Task 2: Implement RNN Model

```bash
# Test model creation
python -c "from src.model import ShakespeareRNN; model = ShakespeareRNN(vocab_size=100, embedding_dim=64, hidden_size=128); print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')"
```

### Task 3: Train Model and Monitor Performance

```bash
# Basic training (quick test)
python -m src.train --epochs 5 --batch_size 16 --seq_len 50

# Full training
python -m src.train --epochs 20 --batch_size 32 --seq_len 100 --outdir outputs
```

### Task 4: Generate Text Predictions

```bash
# Train a model first, then evaluate
python -m src.train --epochs 10 --outdir outputs
python -m src.evaluate --model_path outputs/best_model.pt --num_samples 5 --max_length 50
```

### Task 5: Evaluate Model Performance

```bash
# Comprehensive evaluation
python -m src.evaluate --model_path outputs/best_model.pt --outdir outputs --num_samples 10 --max_length 100
```

## Google Colab Execution (Task 6)

### Setup

1. Open `colab_runner.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Choose one setup option:
   - **Option A**: Mount Google Drive and edit path
   - **Option B**: Clone from GitHub (edit repo URL)
   - **Option C**: Upload ZIP file

### Run All Tasks

Execute all cells in the notebook sequentially. The notebook will:

1. Run ablation study
2. Train optimal model
3. Evaluate performance
4. Generate text samples
5. Create visualizations
6. Download results

## Ablation Study Commands

### Full Ablation Study

```bash
python -m src.ablation_study --outdir outputs --epochs 10
```

### Quick Ablation Test

```bash
python -m src.ablation_study --outdir outputs --epochs 3
```

### Train with Optimal Hyperparameters

```bash
python -m src.train_optimal --outdir outputs --epochs 20 --best_config_file outputs/best_ablation_configs.json
```

## Advanced Usage

### Custom Model Configuration

```bash
python -m src.train \
    --epochs 20 \
    --batch_size 64 \
    --seq_len 150 \
    --embedding_dim 256 \
    --hidden_size 512 \
    --num_layers 3 \
    --dropout 0.2 \
    --rnn_type GRU \
    --lr 0.001
```

### Different Evaluation Settings

```bash
python -m src.evaluate \
    --model_path outputs/optimal_model.pt \
    --batch_size 64 \
    --num_samples 20 \
    --max_length 200 \
    --temperature 0.5
```

### Ablation Study with Custom Settings

```bash
python -m src.ablation_study \
    --outdir outputs \
    --epochs 15 \
    --seq_len 100 \
    --batch_size 32 \
    --vocab_size 200
```

## Output Files

After running the commands, you'll find these files in the `outputs/` directory:

### Training Outputs

- `best_model.pt` - Best model checkpoint
- `optimal_model.pt` - Optimal hyperparameter model
- `training_curves.png` - Training/validation curves
- `training_history.csv` - Training metrics history

### Ablation Study Outputs

- `ablation_results.csv` - All ablation experiment results
- `best_ablation_configs.json` - Best configurations found
- `ablation_study_results.png` - Ablation study visualizations
- `parameter_vs_performance.png` - Model size vs performance plot

### Evaluation Outputs

- `evaluation_metrics.csv` - Comprehensive evaluation metrics
- `generated_texts.txt` - Sample generated texts
- `evaluation_results.png` - Evaluation visualizations

### Comparison Outputs

- `model_comparison.csv` - Optimal vs baseline comparison
- `optimal_vs_baseline.png` - Performance comparison plots
- `text_generation_comparison.txt` - Text generation comparison

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:

   ```bash
   # Reduce batch size
   python -m src.train --batch_size 16
   ```

2. **Slow training**:

   ```bash
   # Reduce sequence length
   python -m src.train --seq_len 50
   ```

3. **Import errors**:

   ```bash
   # Ensure you're in the Q2 directory
   cd Q2
   export PYTHONPATH=$PWD
   ```

4. **Dataset loading issues**:
   ```bash
   # Test dataset loading
   python -c "from datasets import load_dataset; print('Dataset test:', load_dataset('karpathy/tiny_shakespeare'))"
   ```

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Adjust batch size**: Larger batches for better GPU utilization
3. **Sequence length**: Longer sequences for better context but slower training
4. **Model size**: Balance between performance and training time

## Expected Results

### Good Performance Indicators

- **Validation Perplexity**: < 10.0
- **Text Coherence**: Generated text makes sense
- **Training Stability**: Loss decreases smoothly
- **Model Size**: Reasonable parameter count

### Typical Training Times

- **Quick test** (5 epochs): ~2-5 minutes
- **Full training** (20 epochs): ~15-30 minutes
- **Ablation study**: ~1-2 hours
- **Colab execution**: ~30-60 minutes

This command reference should help you run all Q2 tasks efficiently!
