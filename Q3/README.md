# Q3: PixelRNN/CNN Implementation

This project implements and compares three generative models for image data based on the **"Pixel Recurrent Neural Networks"** paper by van den Oord et al. (2016).

## 🎯 **Implemented Models**

### 1. PixelCNN
- **Architecture**: Convolutional neural network with masked convolutions
- **Key Features**:
  - Mask Type A for first layer (excludes center pixel)
  - Mask Type B for subsequent layers (includes center pixel)
  - Residual blocks for deep networks
  - Parallel training due to convolutional structure

### 2. Row LSTM
- **Architecture**: Recurrent neural network processing rows sequentially
- **Key Features**:
  - Input-to-state and state-to-state 1D convolutions
  - Triangular receptive field above current pixel
  - Sequential processing from top to bottom

### 3. Diagonal BiLSTM
- **Architecture**: Bidirectional LSTM processing diagonals
- **Key Features**:
  - Skewing/unskewing operations for diagonal processing
  - Forward and backward LSTM processing
  - Largest receptive field among all models

## 📊 **Dataset**

- **CIFAR-10**: 60,000 32×32 color images in 10 classes
- **Preprocessing**: Discrete pixel values [0, 255] for softmax modeling
- **Splits**: 50,000 training + 10,000 test images

## 🚀 **Quick Start**

### Local Setup
```bash
# Navigate to Q3 directory
cd Q3/

# Test dataset loading
python src/dataset.py

# Test individual models
python src/pixelcnn.py
python src/row_lstm.py
python src/diagonal_bilstm.py
```

### Training Models
```bash
# Train PixelCNN
python -m src.train --model_type pixelcnn --epochs 50 --batch_size 64

# Train Row LSTM
python -m src.train --model_type row_lstm --epochs 30 --batch_size 32

# Train Diagonal BiLSTM
python -m src.train --model_type diagonal_bilstm --epochs 20 --batch_size 16
```

### Evaluation
```bash
# Evaluate all trained models
python -m src.evaluate --model_dir outputs --output_dir evaluation_results
```

### Google Colab
1. Open `colab_runner.ipynb` in Google Colab
2. Enable GPU runtime
3. Run all cells for complete training and evaluation

## 📁 **Project Structure**

```
Q3/
├── src/
│   ├── dataset.py           # CIFAR-10 data loading and preprocessing
│   ├── pixelcnn.py         # PixelCNN implementation with masked convolutions
│   ├── row_lstm.py         # Row LSTM implementation
│   ├── diagonal_bilstm.py  # Diagonal BiLSTM implementation
│   ├── train.py            # Training script for all models
│   ├── evaluate.py         # Evaluation and comparison script
│   └── utils.py            # Utility functions and metrics
├── outputs/                # Training outputs (models, curves, history)
├── evaluation_results/     # Evaluation outputs (comparisons, samples)
├── colab_runner.ipynb     # Google Colab notebook
├── cifar-10-python.tar.gz # CIFAR-10 dataset
└── README.md              # This file
```

## 📈 **Key Metrics**

### Primary Metrics
- **Negative Log-Likelihood (NLL)**: Lower is better
- **Bits per Dimension (BPD)**: Standard generative modeling metric
- **Parameter Count**: Model complexity measure

### Bonus Metrics (Implemented)
- **Inception Score (IS)**: Sample quality measure
- **Fréchet Inception Distance (FID)**: Distribution similarity
- **Visual Quality**: Qualitative sample inspection

## 🔬 **Technical Implementation Details**

### PixelCNN
- **Masked Convolutions**: Ensures autoregressive property
- **Residual Connections**: Enables deep networks (12+ layers)
- **Discrete Softmax**: 256-way classification per pixel channel

### Row LSTM
- **1D Convolutions**: Along width dimension for each row
- **Causal Padding**: Maintains autoregressive ordering
- **Multi-layer Architecture**: Stacked LSTM cells

### Diagonal BiLSTM
- **Skewing Operations**: Transform 2D input for diagonal processing
- **Bidirectional Processing**: Forward and backward along diagonals
- **Complex Architecture**: Most computationally intensive

## 📊 **Expected Results**

Based on the original paper and our implementation:

| Model | Training Speed | Memory Usage | Performance | Receptive Field |
|-------|---------------|--------------|-------------|-----------------|
| PixelCNN | Fast | Low | Good | Limited |
| Row LSTM | Medium | Medium | Better | Triangular |
| Diagonal BiLSTM | Slow | High | Best | Full Context |

## 🛠 **Requirements**

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.3.0
pandas>=1.3.0
seaborn>=0.11.0
```

## 🎓 **Academic Context**

This implementation follows the **"Pixel Recurrent Neural Networks"** paper:
- **Authors**: Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu
- **Published**: ICML 2016
- **Key Contributions**:
  - Autoregressive pixel-by-pixel image generation
  - Novel architectures for capturing spatial dependencies
  - State-of-the-art results on image modeling benchmarks

## 🏆 **Assignment Completion**

### ✅ Completed Tasks
1. **Paper Understanding**: Implemented all three architectures from Sections 2-3
2. **Model Implementation**:
   - ✅ PixelCNN with masked convolutions (Type A/B)
   - ✅ Row LSTM with input-to-state/state-to-state convolutions
   - ✅ Diagonal BiLSTM with skewing/unskewing operations
3. **Training**: All models trained on CIFAR-10 with discrete softmax
4. **Evaluation**: NLL and bits/dimension metrics implemented
5. **Comparison**: Comprehensive model comparison with visualizations
6. **Bonus Metrics**: IS, FID, and visual quality assessment

### 📋 **Deliverables**
- ✅ Complete implementations of all three models
- ✅ Training scripts with proper monitoring
- ✅ Evaluation framework with standard metrics
- ✅ Google Colab notebook for easy execution
- ✅ Comprehensive comparison and analysis
- ✅ Generated sample visualizations

## 🚀 **Running on Google Colab**

1. **Upload**: Use the provided `colab_runner.ipynb` notebook
2. **Setup**: Enable GPU runtime for faster training
3. **Data**: Upload `cifar-10-python.tar.gz` or clone from GitHub
4. **Execute**: Run all cells for complete pipeline
5. **Download**: Results automatically packaged for download

## 📝 **Notes**

- **Memory Requirements**: Diagonal BiLSTM requires significant GPU memory
- **Training Time**: Full training can take several hours on GPU
- **Hyperparameters**: Configurations optimized for demonstration
- **Reproducibility**: Random seeds set for consistent results

---

**Implementation by**: Mahad  
**Course**: GenAI Assignment Q3  
**Based on**: "Pixel Recurrent Neural Networks" (van den Oord et al., 2016)