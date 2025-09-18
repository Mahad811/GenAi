import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RowLSTMCell(nn.Module):
    """
    Row LSTM Cell with input-to-state and state-to-state convolutions
    
    Processes one row at a time with triangular receptive field
    """
    
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super(RowLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        # Input-to-state convolution (1D convolution along width)
        # Uses causal padding to maintain autoregressive property
        self.input_conv = nn.Conv1d(
            input_channels, 4 * hidden_channels, kernel_size,
            padding=kernel_size - 1, bias=True
        )
        
        # State-to-state convolution (1D convolution along width)
        self.hidden_conv = nn.Conv1d(
            hidden_channels, 4 * hidden_channels, kernel_size,
            padding=kernel_size - 1, bias=False
        )
        
    def forward(self, input_row: torch.Tensor, 
                hidden_state: Optional[torch.Tensor] = None,
                cell_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one row
        
        Args:
            input_row: (B, C, W) - one row of the image
            hidden_state: (B, H, W) - previous hidden state
            cell_state: (B, H, W) - previous cell state
            
        Returns:
            output: (B, H, W) - output for this row
            new_hidden: (B, H, W) - new hidden state
            new_cell: (B, H, W) - new cell state
        """
        batch_size, input_channels, width = input_row.shape
        
        # Initialize states if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_channels, width,
                                     device=input_row.device, dtype=input_row.dtype)
        if cell_state is None:
            cell_state = torch.zeros(batch_size, self.hidden_channels, width,
                                   device=input_row.device, dtype=input_row.dtype)
        
        # Input-to-state transformation
        input_transform = self.input_conv(input_row)
        # Remove extra padding to maintain width
        input_transform = input_transform[:, :, :width]
        
        # State-to-state transformation
        hidden_transform = self.hidden_conv(hidden_state)
        # Remove extra padding to maintain width
        hidden_transform = hidden_transform[:, :, :width]
        
        # Combined transformation
        combined = input_transform + hidden_transform
        
        # Split into gates: input, forget, output, candidate
        i_gate, f_gate, o_gate, g_gate = torch.split(combined, self.hidden_channels, dim=1)
        
        # Apply gate activations
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)
        
        # Update cell state
        new_cell = f_gate * cell_state + i_gate * g_gate
        
        # Update hidden state
        new_hidden = o_gate * torch.tanh(new_cell)
        
        return new_hidden, new_hidden, new_cell


class RowLSTM(nn.Module):
    """
    Row LSTM for autoregressive image generation
    
    Based on "Pixel Recurrent Neural Networks" by van den Oord et al.
    Processes the image row by row from top to bottom.
    """
    
    def __init__(self, input_channels: int = 3, hidden_channels: int = 128,
                 num_layers: int = 7, num_classes: int = 256):
        super(RowLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        
        # Row LSTM layers
        self.lstm_layers = nn.ModuleList([
            RowLSTMCell(hidden_channels if i == 0 else hidden_channels, 
                       hidden_channels, kernel_size=3)
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, input_channels * num_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (B, C, H, W) with pixel values [0, 255]
            
        Returns:
            logits: (B, C, num_classes, H, W) logits for each pixel and color channel
        """
        batch_size, channels, height, width = x.shape
        
        # Convert to float and normalize to [0, 1]
        x = x.float() / 255.0
        
        # Input projection
        x = self.input_proj(x)
        
        # Initialize states for all layers
        hidden_states = [None] * self.num_layers
        cell_states = [None] * self.num_layers
        
        # Process row by row
        output_rows = []
        
        for row_idx in range(height):
            current_row = x[:, :, row_idx, :]  # (B, H, W)
            
            # Pass through LSTM layers
            for layer_idx, lstm_layer in enumerate(self.lstm_layers):
                current_row, hidden_states[layer_idx], cell_states[layer_idx] = \
                    lstm_layer(current_row, hidden_states[layer_idx], cell_states[layer_idx])
            
            output_rows.append(current_row)
        
        # Stack rows back together
        output = torch.stack(output_rows, dim=2)  # (B, H, H, W)
        
        # Output projection
        output = self.output_proj(output)
        
        # Reshape to (B, C, num_classes, H, W)
        output = output.view(batch_size, channels, self.num_classes, height, width)
        
        return output
    
    def sample(self, shape: tuple, device: torch.device, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate samples using autoregressive sampling
        
        Args:
            shape: (batch_size, channels, height, width)
            device: Device to generate on
            temperature: Sampling temperature
            
        Returns:
            Generated images (B, C, H, W) with values [0, 255]
        """
        batch_size, channels, height, width = shape
        
        # Initialize with zeros
        samples = torch.zeros(shape, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    for c in range(channels):
                        # Get logits for current pixel
                        logits = self.forward(samples)
                        logits = logits[:, c, :, i, j] / temperature
                        
                        # Sample from categorical distribution
                        probs = F.softmax(logits, dim=-1)
                        pixel_value = torch.multinomial(probs, 1).squeeze(-1)
                        
                        # Update the sample
                        samples[:, c, i, j] = pixel_value
        
        return samples
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Row LSTM',
            'input_channels': self.input_channels,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


if __name__ == '__main__':
    # Test Row LSTM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RowLSTM(input_channels=3, hidden_channels=64, num_layers=4).to(device)
    
    # Test forward pass
    batch_size = 2
    x = torch.randint(0, 256, (batch_size, 3, 16, 16), dtype=torch.long).to(device)
    
    print("Testing Row LSTM...")
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input range: [{x.min()}, {x.max()}]")
    
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: (batch_size, channels, num_classes, height, width)")
    
    # Test model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test Row LSTM Cell
    print(f"\nTesting Row LSTM Cell...")
    cell = RowLSTMCell(input_channels=64, hidden_channels=64)
    input_row = torch.randn(2, 64, 16)  # (B, C, W)
    
    output, hidden, cell_state = cell(input_row)
    print(f"Cell input shape: {input_row.shape}")
    print(f"Cell output shape: {output.shape}")
    print(f"Cell hidden shape: {hidden.shape}")
    print(f"Cell state shape: {cell_state.shape}")