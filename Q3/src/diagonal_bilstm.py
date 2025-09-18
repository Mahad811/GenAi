import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def skew_input(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Skew the input tensor for diagonal processing
    
    Args:
        input_tensor: (B, C, H, W) input tensor
        
    Returns:
        skewed_tensor: (B, C, H, W + H - 1) skewed tensor
    """
    batch_size, channels, height, width = input_tensor.shape
    
    # Create skewed tensor with extra width
    skewed = torch.zeros(batch_size, channels, height, width + height - 1,
                        device=input_tensor.device, dtype=input_tensor.dtype)
    
    for i in range(height):
        # Each row is shifted by its index
        skewed[:, :, i, i:i + width] = input_tensor[:, :, i, :]
    
    return skewed


def unskew_output(skewed_tensor: torch.Tensor, original_width: int) -> torch.Tensor:
    """
    Unskew the output tensor back to original dimensions
    
    Args:
        skewed_tensor: (B, C, H, W + H - 1) skewed tensor
        original_width: Original width of the tensor
        
    Returns:
        output_tensor: (B, C, H, W) unskewed tensor
    """
    batch_size, channels, height, _ = skewed_tensor.shape
    
    # Create output tensor with original dimensions
    output = torch.zeros(batch_size, channels, height, original_width,
                        device=skewed_tensor.device, dtype=skewed_tensor.dtype)
    
    for i in range(height):
        # Extract the original row from skewed position
        output[:, :, i, :] = skewed_tensor[:, :, i, i:i + original_width]
    
    return output


class DiagonalLSTMCell(nn.Module):
    """
    Diagonal LSTM Cell for processing diagonals in skewed input
    """
    
    def __init__(self, input_channels: int, hidden_channels: int):
        super(DiagonalLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Input-to-state transformation
        self.input_transform = nn.Linear(input_channels, 4 * hidden_channels, bias=True)
        
        # State-to-state transformation
        self.hidden_transform = nn.Linear(hidden_channels, 4 * hidden_channels, bias=False)
        
    def forward(self, input_step: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None,
                cell_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one step along diagonal
        
        Args:
            input_step: (B, C) - input for current step
            hidden_state: (B, H) - previous hidden state
            cell_state: (B, H) - previous cell state
            
        Returns:
            output: (B, H) - output for this step
            new_hidden: (B, H) - new hidden state
            new_cell: (B, H) - new cell state
        """
        batch_size = input_step.shape[0]
        
        # Initialize states if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_channels,
                                     device=input_step.device, dtype=input_step.dtype)
        if cell_state is None:
            cell_state = torch.zeros(batch_size, self.hidden_channels,
                                   device=input_step.device, dtype=input_step.dtype)
        
        # Input and hidden transformations
        input_transform = self.input_transform(input_step)
        hidden_transform = self.hidden_transform(hidden_state)
        
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


class DiagonalBiLSTM(nn.Module):
    """
    Diagonal BiLSTM for autoregressive image generation
    
    Based on "Pixel Recurrent Neural Networks" by van den Oord et al.
    Processes the image along diagonals in both directions.
    """
    
    def __init__(self, input_channels: int = 3, hidden_channels: int = 128,
                 num_layers: int = 7, num_classes: int = 256):
        super(DiagonalBiLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        
        # Forward and backward LSTM layers
        self.forward_lstms = nn.ModuleList([
            DiagonalLSTMCell(hidden_channels if i == 0 else hidden_channels * 2,
                           hidden_channels)
            for i in range(num_layers)
        ])
        
        self.backward_lstms = nn.ModuleList([
            DiagonalLSTMCell(hidden_channels if i == 0 else hidden_channels * 2,
                           hidden_channels)
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels * 2, input_channels * num_classes, kernel_size=1)
        
    def process_diagonals(self, skewed_input: torch.Tensor) -> torch.Tensor:
        """
        Process diagonals with bidirectional LSTM
        
        Args:
            skewed_input: (B, C, H, W + H - 1) skewed input
            
        Returns:
            processed: (B, C*2, H, W + H - 1) processed output
        """
        batch_size, channels, height, skewed_width = skewed_input.shape
        
        # Initialize outputs
        forward_outputs = torch.zeros(batch_size, self.hidden_channels, height, skewed_width,
                                    device=skewed_input.device, dtype=skewed_input.dtype)
        backward_outputs = torch.zeros(batch_size, self.hidden_channels, height, skewed_width,
                                     device=skewed_input.device, dtype=skewed_input.dtype)
        
        # Process each diagonal
        for diag_idx in range(height + skewed_width - 1):
            # Get diagonal positions
            positions = []
            for row in range(height):
                col = diag_idx - row
                if 0 <= col < skewed_width:
                    positions.append((row, col))
            
            if not positions:
                continue
            
            # Extract diagonal values
            diagonal_values = []
            for row, col in positions:
                diagonal_values.append(skewed_input[:, :, row, col])
            
            if diagonal_values:
                diagonal_tensor = torch.stack(diagonal_values, dim=1)  # (B, num_positions, C)
                
                # Process forward direction
                forward_states = [None] * self.num_layers
                forward_cells = [None] * self.num_layers
                
                for step in range(len(positions)):
                    current_input = diagonal_tensor[:, step, :]  # (B, C)
                    
                    for layer_idx, lstm_layer in enumerate(self.forward_lstms):
                        if layer_idx == 0:
                            lstm_input = current_input
                        else:
                            # Concatenate forward and backward from previous layer
                            prev_forward = forward_outputs[:, :, positions[step][0], positions[step][1]]
                            prev_backward = backward_outputs[:, :, positions[step][0], positions[step][1]]
                            lstm_input = torch.cat([prev_forward, prev_backward], dim=1)
                        
                        output, forward_states[layer_idx], forward_cells[layer_idx] = \
                            lstm_layer(lstm_input, forward_states[layer_idx], forward_cells[layer_idx])
                        
                        if layer_idx == self.num_layers - 1:
                            row, col = positions[step]
                            forward_outputs[:, :, row, col] = output
                
                # Process backward direction
                backward_states = [None] * self.num_layers
                backward_cells = [None] * self.num_layers
                
                for step in reversed(range(len(positions))):
                    current_input = diagonal_tensor[:, step, :]  # (B, C)
                    
                    for layer_idx, lstm_layer in enumerate(self.backward_lstms):
                        if layer_idx == 0:
                            lstm_input = current_input
                        else:
                            # Concatenate forward and backward from previous layer
                            prev_forward = forward_outputs[:, :, positions[step][0], positions[step][1]]
                            prev_backward = backward_outputs[:, :, positions[step][0], positions[step][1]]
                            lstm_input = torch.cat([prev_forward, prev_backward], dim=1)
                        
                        output, backward_states[layer_idx], backward_cells[layer_idx] = \
                            lstm_layer(lstm_input, backward_states[layer_idx], backward_cells[layer_idx])
                        
                        if layer_idx == self.num_layers - 1:
                            row, col = positions[step]
                            backward_outputs[:, :, row, col] = output
        
        # Combine forward and backward outputs
        combined = torch.cat([forward_outputs, backward_outputs], dim=1)
        
        return combined
    
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
        
        # Skew input for diagonal processing
        skewed = skew_input(x)
        
        # Process diagonals
        processed = self.process_diagonals(skewed)
        
        # Unskew output
        output = unskew_output(processed, width)
        
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
            'model_type': 'Diagonal BiLSTM',
            'input_channels': self.input_channels,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


if __name__ == '__main__':
    # Test Diagonal BiLSTM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test skewing and unskewing
    print("Testing skew/unskew operations...")
    test_input = torch.randn(1, 3, 4, 4)
    print(f"Original shape: {test_input.shape}")
    
    skewed = skew_input(test_input)
    print(f"Skewed shape: {skewed.shape}")
    
    unskewed = unskew_output(skewed, 4)
    print(f"Unskewed shape: {unskewed.shape}")
    
    # Check if unskewing recovers original
    print(f"Skew/unskew preserves data: {torch.allclose(test_input, unskewed)}")
    
    # Test model
    model = DiagonalBiLSTM(input_channels=3, hidden_channels=32, num_layers=2).to(device)
    
    # Test forward pass (small image for memory)
    batch_size = 1
    x = torch.randint(0, 256, (batch_size, 3, 8, 8), dtype=torch.long).to(device)
    
    print(f"\nTesting Diagonal BiLSTM...")
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