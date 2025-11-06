import torch
import torch.nn as nn

class LinearValueNetwork(nn.Module):
    """Linear value network for state value estimation."""
    
    def __init__(self, state_dim: int):
        super().__init__()
        self.linear = nn.Linear(state_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state).squeeze(-1)

class MLPValueNetwork(nn.Module):
    """MLP value network for state value estimation."""
    
    def __init__(self, state_dim: int, hidden_dims: list[int] = [128, 64]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)
