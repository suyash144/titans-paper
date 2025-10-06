import torch
from torch import Tensor
import torch.nn.functional as F
from neural_memory import NeuralMemory


class MACLayer(torch.nn.Module):
    """Memory as Context (MAC) layer using Neural Memory"""
    
    def __init__(self, hidden_dim: int, sequence_len: int, persist_mem_len: int, mem_network_layers: int = 2, 
                 alpha: float = 0.999, eta: float = 0.8, theta: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.persist_mem_len = persist_mem_len
        self.combined_dim = persist_mem_len + 2 * sequence_len
        
        # Persistent memory weights
        self.persist_mem_weights = torch.nn.Parameter(
            torch.randn(persist_mem_len, hidden_dim)
        )
        
        # Attention part
        self.attention_processor = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation=F.silu,
            batch_first=True
        )
        
        # Query projection for memory retrieval
        self.query_projection = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # Instantiate a neural memory module
        self.memory = NeuralMemory(
            dimension=hidden_dim,
            layers=mem_network_layers,
            intermediate_dim=2 * hidden_dim,
            momentum=alpha,
            surprise_decay=eta,
            learn_rate=theta
        )
        
        # Final transformation
        self.final_transform = torch.nn.Linear(
            self.combined_dim * hidden_dim,
            sequence_len * hidden_dim
        )
        
        # Activation and gating
        self.act_silu = torch.nn.SiLU()
        self.gate_sigmoid = torch.nn.Sigmoid()
        
        # Track outer parameters
        self.outer_param_list = (
            [self.persist_mem_weights] + 
            list(self.query_projection.parameters()) + 
            list(self.final_transform.parameters()) + 
            list(self.attention_processor.parameters())
        )
            
    def forward(self, x_input: Tensor) -> Tensor:
        batch_count = x_input.shape[0]
        
        # Generate queries and retrieve from gradient memory
        flat_queries = F.normalize(self.act_silu(
            self.query_projection(x_input.reshape(-1, self.hidden_dim))
        ))
        retrieved_mem = self.memory.lookup(flat_queries)
        retrieved_mem = retrieved_mem.view(batch_count, -1, self.hidden_dim)
        
        # Expand persistent memory for batch
        persist_expanded = self.persist_mem_weights.unsqueeze(0).repeat(batch_count, 1, 1)
        
        # Concatenate: [persistent_mem, retrieved_mem, input]
        concatenated = torch.cat([persist_expanded, retrieved_mem, x_input], dim=1)
        
        # Pass through attention
        attention_out = self.attention_processor(concatenated)
        attention_out = self.act_silu(attention_out.reshape(-1, self.combined_dim * self.hidden_dim))
        
        # Final transformation
        transformed = self.final_transform(attention_out).view(-1, self.hidden_dim)
        
        # Update gradient memory
        _, updated_weights = self.memory.update(transformed)
        
        # Retrieve with updated parameters
        updated_retrieval = torch.func.functional_call(
            self.memory,
            updated_weights,
            F.normalize(self.act_silu(self.query_projection(transformed)))
        )
        
        # Gating output
        gated_output = transformed * self.gate_sigmoid(updated_retrieval)
        
        return gated_output.view(batch_count, self.sequence_len, self.hidden_dim)


class MAC(torch.nn.Module):
    """Memory as Context (MAC) architecture"""
    
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, ctx_window: int, persist_mem: int, num_layers: int = 2,
                mem_layers: int = 2, alpha: float = 0.999, eta: float = 0.8, theta: float = 0.3):
        super().__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.ctx_window = ctx_window
        
        # Input embedding
        self.input_embedding = torch.nn.Linear(in_dim, hid_dim)
        
        # Stack of MAC layers
        self.layer_stack = torch.nn.ModuleList([
            MACLayer(
                hid_dim, ctx_window, persist_mem,
                mem_network_layers=mem_layers,
                alpha=alpha,
                eta=eta,
                theta=theta
            )
            for _ in range(num_layers)
        ])

        self.output_projection = torch.nn.Linear(hid_dim * ctx_window, out_dim)
        self.act = torch.nn.SiLU()
        self.trainable_params = list(self.input_embedding.parameters())
        self.trainable_params += list(self.output_projection.parameters())
        for layer in self.layer_stack:
            self.trainable_params.extend(layer.outer_param_list)
    
    def chunk_forward(self, chunk: Tensor) -> Tensor:
        """Process single chunk through all layers"""
        batch_sz = chunk.shape[0]
        
        # Embed input
        embedded = self.input_embedding(chunk.reshape(-1, self.in_dim))
        embedded = embedded.view(batch_sz, self.ctx_window, self.hid_dim)
        
        # Process through layer stack with residuals
        for layer in self.layer_stack:
            layer_out = layer(embedded)
            embedded = embedded + self.act(layer_out)
        
        # Project to output
        output = self.output_projection(embedded.view(batch_sz, -1))
        return output
    
    def forward(self, sequences: Tensor) -> Tensor:
        """Process full sequence using windowed approach"""
        batch_sz, seq_length, _ = sequences.shape
        device = sequences.device
        
        # Initialise output tensor
        output_tensor = torch.zeros(batch_sz, seq_length, self.out_dim, device=device)
        
        # Handle initial positions that don't have full context (positions 0 to ctx_window-2)
        for pos in range(min(self.ctx_window - 1, seq_length)):
            # Take sequence from start to current position + 1
            partial_seq = sequences[:, :pos + 1]
            
            # Pad to ctx_window size
            padding_size = self.ctx_window - partial_seq.shape[1]
            if padding_size > 0:
                padding = torch.zeros(batch_sz, padding_size, self.in_dim, device=device)
                padded_seq = torch.cat([padding, partial_seq], dim=1)
            else:
                padded_seq = partial_seq
            
            # Process and assign
            output = self.chunk_forward(padded_seq)
            output_tensor[:, pos] = output
        
        # Process full windows for positions >= ctx_window-1
        if seq_length >= self.ctx_window:
            unfolded = sequences.unfold(1, self.ctx_window, 1)
            unfolded = unfolded.permute(0, 1, 3, 2)
            n_windows = unfolded.shape[1]
            
            for offset in range(self.ctx_window):
                if offset >= n_windows:
                    break
                
                chunk_indices = list(range(offset, n_windows, self.ctx_window))
                if not chunk_indices:
                    continue
                
                chunks = unfolded[:, chunk_indices].reshape(-1, self.ctx_window, self.in_dim)
                chunk_outputs = self.chunk_forward(chunks)
                chunk_outputs = chunk_outputs.view(batch_sz, len(chunk_indices), self.out_dim)
                
                for i, chunk_idx in enumerate(chunk_indices):
                    output_pos = chunk_idx + self.ctx_window - 1
                    if output_pos < seq_length:
                        output_tensor[:, output_pos] = chunk_outputs[:, i]
        
        return output_tensor