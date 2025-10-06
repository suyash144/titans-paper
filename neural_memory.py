import torch
from torch import Tensor
from typing import Tuple, Dict
import torch.nn.functional as F


class NeuralMemory(torch.nn.Module):
    """Neural long-term memory module"""
    
    def __init__(self, dimension: int = 16, layers: int = 2, intermediate_dim: int = 32, momentum: float = 0.9, surprise_decay: float = 0.60, learn_rate: float = 0.1):
        super().__init__()
        
        self.dim = dimension
        self.momentum_factor = momentum
        self.surprise_decay = surprise_decay
        self.learn_rate = learn_rate
        
        self.memory_network = torch.nn.ModuleList()
        
        if layers == 1:
            self.memory_network.append(
                torch.nn.Linear(dimension, dimension)
            )
        else:
            self.memory_network.append(torch.nn.Sequential(
                torch.nn.Linear(dimension, intermediate_dim),
                torch.nn.SiLU()
            ))

            for _ in range(layers - 2):
                self.memory_network.append(torch.nn.Sequential(
                    torch.nn.Linear(intermediate_dim, intermediate_dim),
                    torch.nn.SiLU()
                ))

            self.memory_network.append(
                torch.nn.Linear(intermediate_dim, dimension)
            )
        
        # Key, query and value transformations
        self.key_transform = torch.nn.Linear(dimension, dimension, bias=False)
        self.val_transform = torch.nn.Linear(dimension, dimension, bias=False)
        
        # Initialise with Xavier
        torch.nn.init.xavier_uniform_(self.key_transform.weight.data)
        torch.nn.init.xavier_uniform_(self.val_transform.weight.data)
        
        self.act_fn = torch.nn.SiLU()
        self.surprise_state = {}
    
    def lookup(self, query: Tensor) -> Tensor:
        """Retrieve from memory using current parameters"""
        output = query
        for layer in self.memory_network:
            output = layer(output)
        return output
    
    def forward(self, query: Tensor) -> Tensor:
        """Forward method for compatibility with torch.func.functional_call"""
        return self.lookup(query)
    
    def update(self, data: Tensor) -> Tuple[float, Dict[str, Tensor]]:
        """Update memory using gradient-based learning with surprise"""

        with torch.enable_grad():
            data_detached = data.detach()
            
            # Transform to keys and values
            key_vectors = F.normalize(self.act_fn(self.key_transform(data_detached)))
            val_vectors = self.act_fn(self.val_transform(data_detached))
            
            # Forward pass through memory network
            memory_output = key_vectors
            for layer in self.memory_network:
                memory_output = layer(memory_output)
            
            # Compute reconstruction error
            error = ((memory_output - val_vectors)**2).mean(axis=0).sum()
            
            grad_list = torch.autograd.grad(error, self.parameters(), create_graph=False, allow_unused=True)
            
            # Update parameters with surprise mechanism
            new_params = {}
            param_items = list(self.named_parameters())
            trainable_param_names = [name for name, p in param_items if p.requires_grad]
            
            # Initialize new_params with all current parameter values
            for param_name, param_tensor in param_items:
                new_params[param_name] = param_tensor.data
            
            # Update only trainable parameters that have gradients
            for param_name, grad_tensor in zip(trainable_param_names, grad_list):
                if grad_tensor is not None:
                    param_tensor = dict(self.named_parameters())[param_name]
                    
                    # Initialise surprise if needed
                    if param_name not in self.surprise_state:
                        self.surprise_state[param_name] = torch.zeros_like(grad_tensor)
                    
                    # Update surprise state
                    self.surprise_state[param_name] = (
                        self.surprise_state[param_name] * self.surprise_decay - 
                        self.learn_rate * grad_tensor
                    )
                    
                    # Update parameters (skip K and V transforms as these are not learned in the inner loop)
                    if param_name.startswith('key_transform') or param_name.startswith('val_transform'):
                        new_params[param_name] = param_tensor.data
                    else:
                        new_params[param_name] = (
                            self.momentum_factor * param_tensor.data + 
                            self.surprise_state[param_name]
                        )
                    
                    # Apply update
                    param_tensor.data = new_params[param_name]
        
        return error.item(), new_params