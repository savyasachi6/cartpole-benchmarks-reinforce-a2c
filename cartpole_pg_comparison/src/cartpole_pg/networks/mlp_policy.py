import torch
import torch.nn as nn
from torch.distributions import Categorical

class MLPPolicy(nn.Module):
    """Multilayer perceptron policy network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int] = [128, 64]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> Categorical:
        logits = self.network(state)
        return Categorical(logits=logits)
    
    def get_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
