import torch
import torch.nn as nn
from torch.distributions import Categorical

class LinearPolicy(nn.Module):
    """Linear policy network with softmax output for discrete actions."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(state_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Categorical:
        """
        Compute action distribution.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Categorical distribution over actions
        """
        logits = self.linear(state)
        return Categorical(logits=logits)
    
    def get_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Sample action and return log probability."""
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy for given state-action pairs."""
        dist = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
