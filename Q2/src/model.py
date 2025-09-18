import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.nn.functional as F


class ShakespeareRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_size: int = 256, num_layers: int = 2, 
                 dropout: float = 0.3, rnn_type: str = 'LSTM'):
        """
        RNN model for next-word prediction on Shakespeare text
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_size: Hidden state size
            num_layers: Number of RNN layers
            dropout: Dropout rate
            rnn_type: Type of RNN ('LSTM', 'GRU', 'RNN')
        """
        super(ShakespeareRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:  # RNN
            self.rnn = nn.RNN(
                embedding_dim, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x: torch.Tensor, hidden = None):
        """
        Forward pass
        
        Args:
            x: Input sequences [batch_size, seq_len]
            hidden: Previous hidden state
            
        Returns:
            output: Logits [batch_size, seq_len, vocab_size]
            hidden: New hidden state
        """
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # RNN
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        rnn_out, hidden = self.rnn(embedded, hidden)
        
        # Apply dropout
        rnn_out = self.dropout(rnn_out)
        
        # Output layer
        output = self.fc(rnn_out)  # [batch_size, seq_len, vocab_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state"""
        if self.rnn_type == 'LSTM':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            return (h0, c0)
        else:  # GRU or RNN
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            return h0
    
    def generate(self, idx_to_char: dict, char_to_idx: dict, 
                seed_text: str, max_length: int = 100, 
                temperature: float = 1.0, device: torch.device = None) -> str:
        """
        Generate text from seed
        
        Args:
            idx_to_char: Index to character mapping
            char_to_idx: Character to index mapping
            seed_text: Starting text
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            device: Device to run on
            
        Returns:
            Generated text
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
            
        generated = seed_text
        hidden = None
        
        # Convert seed to indices
        seed_indices = [char_to_idx.get(char, 0) for char in seed_text]
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                input_seq = torch.tensor([seed_indices[-1]], dtype=torch.long).unsqueeze(0).to(device)
                
                # Forward pass
                output, hidden = self.forward(input_seq, hidden)
                
                # Get logits for last character
                logits = output[0, -1, :] / temperature
                
                # Sample next character
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                
                # Add to generated text
                next_char = idx_to_char.get(next_idx, '')
                generated += next_char
                seed_indices.append(next_idx)
                
                # Stop if we hit a natural stopping point
                if next_char in ['.', '!', '?'] and len(generated) > len(seed_text) + 10:
                    break
        
        return generated
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'rnn_type': self.rnn_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class ShakespeareTransformer(nn.Module):
    """
    Alternative Transformer-based model for comparison
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 num_heads: int = 8, num_layers: int = 4, 
                 max_seq_len: int = 100, dropout: float = 0.1):
        super(ShakespeareTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
        embedded = token_emb + pos_emb
        
        # Transformer
        transformer_out = self.transformer(embedded)
        
        # Output
        output = self.dropout(transformer_out)
        output = self.fc(output)
        
        return output


if __name__ == '__main__':
    # Test the model
    print("Testing Shakespeare RNN Model...")
    
    vocab_size = 100
    model = ShakespeareRNN(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        rnn_type='LSTM'
    )
    
    # Test forward pass
    batch_size, seq_len = 4, 50
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output, hidden = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state type: {type(hidden)}")
    
    # Test model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test text generation
    idx_to_char = {i: chr(65 + i) for i in range(26)}  # A-Z
    char_to_idx = {v: k for k, v in idx_to_char.items()}
    
    generated = model.generate(
        idx_to_char=idx_to_char,
        char_to_idx=char_to_idx,
        seed_text="Hello",
        max_length=20,
        temperature=0.8
    )
    print(f"\nGenerated text: '{generated}'")