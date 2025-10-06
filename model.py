import torch
from torch import Tensor
import torch.nn.functional as F


class StandardTransformerModel(torch.nn.Module):
    """Standard transformer with windowed processing"""
    
    def __init__(self, input_features: int, embed_features: int, context_len: int, num_blocks: int = 2, out_dim: int = 10):
        super().__init__()
        self.context_len = context_len
        self.embed_features = embed_features
        self.out_dim = out_dim
        
        # Input projection
        self.input_embed = torch.nn.Linear(input_features, embed_features)
        
        # Standard transformer encoder blocks
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_features,
            nhead=max(1, embed_features // 64) * 2,  # Ensure even number of heads
            dim_feedforward=embed_features * 4,
            dropout=0.0,
            activation=F.silu,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_blocks
        )
        
        # Output projection - changed to output vocab_size per position
        self.output_head = torch.nn.Linear(embed_features, out_dim)
        
        # Add trainable_params for optimizer compatibility
        self.trainable_params = list(self.parameters())
    
    def forward(self, inputs: Tensor) -> Tensor:
        batch_dim, sequence_len, feat_dim = inputs.shape
        device = inputs.device
        
        # Embed inputs
        embedded = self.input_embed(inputs.view(-1, feat_dim))
        embedded = embedded.view(batch_dim, sequence_len, self.embed_features)
        
        # Create output tensor
        result = torch.zeros(batch_dim, sequence_len, self.out_dim, device=device)
        
        # Handle initial positions with padding
        for pos in range(min(self.context_len, sequence_len)):

            partial_seq = embedded[:, :pos + 1]
            
            # Pad to context_len size
            padding_size = self.context_len - partial_seq.shape[1]
            if padding_size > 0:
                padding = torch.zeros(batch_dim, padding_size, self.embed_features, device=device)
                padded_seq = torch.cat([padding, partial_seq], dim=1)
            else:
                padded_seq = partial_seq
            
            # Apply transformer
            encoded = self.transformer_encoder(padded_seq)
            
            # Output for last position
            result[:, pos, :] = self.output_head(encoded[:, -1, :])
        
        # Process full windows
        for end_idx in range(self.context_len, sequence_len + 1):
            start_idx = end_idx - self.context_len
            window = embedded[:, start_idx:end_idx, :]
            
            # Apply transformer
            encoded = self.transformer_encoder(window)
            
            # Output for last position
            result[:, end_idx - 1, :] = self.output_head(encoded[:, -1, :])
        
        return result


class TransformerWithStaticMemory(torch.nn.Module):
    """Transformer with static memory tokens"""
    
    def __init__(self, in_features: int, hidden_features: int, seq_length: int, static_mem_size: int, n_blocks: int = 2, out_dim: int = 10):
        super().__init__()
        self.seq_length = seq_length
        self.static_size = static_mem_size
        self.hidden_features = hidden_features
        self.out_dim = out_dim
        
        # Static memory tokens (persistent memory)
        self.static_mem_params = torch.nn.Parameter(
            torch.zeros(static_mem_size, hidden_features)
        )
        torch.nn.init.normal_(self.static_mem_params, std=0.02)
        
        # Feature embedding
        self.feature_embedding = torch.nn.Linear(in_features, hidden_features)
        
        # Transformer blocks
        encoder_config = torch.nn.TransformerEncoderLayer(
            d_model=hidden_features,
            nhead=max(1, hidden_features // 32) * 2,
            dim_feedforward=hidden_features * 4,
            dropout=0.0,
            activation=F.silu,
            batch_first=True
        )
        self.transformer_blocks = torch.nn.TransformerEncoder(encoder_config, n_blocks)
        
        # Final projection - changed to output per position
        self.final_proj = torch.nn.Linear(hidden_features, out_dim)
        
        # Add trainable_params for optimizer compatibility
        self.trainable_params = list(self.parameters())
        
        self._device = None
    
    def forward(self, inputs: Tensor) -> Tensor:
        n_samples, full_length, _ = inputs.shape
        device = inputs.device
        
        # Embed input features
        embeddings = self.feature_embedding(inputs.reshape(-1, inputs.shape[-1]))
        embeddings = embeddings.reshape(n_samples, full_length, self.hidden_features)
        
        # Process with sliding windows
        predictions = torch.zeros(n_samples, full_length, self.out_dim, device=device)
        
        # Handle initial positions with padding
        for pos in range(min(self.seq_length, full_length)):
            partial_seq = embeddings[:, :pos + 1, :]
            
            # Pad to seq_length
            padding_size = self.seq_length - partial_seq.shape[1]
            if padding_size > 0:
                padding = torch.zeros(n_samples, padding_size, self.hidden_features, device=device)
                padded_seq = torch.cat([padding, partial_seq], dim=1)
            else:
                padded_seq = partial_seq
            
            # Attach static memory
            static_broadcast = self.static_mem_params[None, :, :].repeat(n_samples, 1, 1)
            combined_seq = torch.cat([static_broadcast, padded_seq], dim=1)
            
            # Transform
            transformed = self.transformer_blocks(combined_seq)
            
            # Output for last position (skip static memory tokens)
            predictions[:, pos, :] = self.final_proj(transformed[:, self.static_size + pos, :])
        
        # Process full windows
        for pos in range(self.seq_length, full_length + 1):
            window_start = pos - self.seq_length
            current_window = embeddings[:, window_start:pos, :]
            
            # Attach static memory to window
            static_broadcast = self.static_mem_params[None, :, :].repeat(n_samples, 1, 1)
            combined_seq = torch.cat([static_broadcast, current_window], dim=1)
            
            # Transform the combined sequence
            transformed = self.transformer_blocks(combined_seq)
            
            # Generate prediction for last position
            predictions[:, pos - 1, :] = self.final_proj(transformed[:, -1, :])
        
        return predictions