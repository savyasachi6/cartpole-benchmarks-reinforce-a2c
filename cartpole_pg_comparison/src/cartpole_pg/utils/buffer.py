import torch
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Transition:
    """Single transition in the environment."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: torch.Tensor

class EpisodeBuffer:
    """Buffer for storing complete episodes."""
    
    def __init__(self):
        self.transitions: List[Transition] = []
        
    def add(self, transition: Transition):
        self.transitions.append(transition)
        
    def clear(self):
        self.transitions = []
        
    def compute_returns(self, gamma: float) -> torch.Tensor:
        """Compute discounted returns (Monte Carlo)."""
        returns = []
        G = 0
        
        for t in reversed(self.transitions):
            G = t.reward + gamma * G
            returns.insert(0, G)
            
        return torch.tensor(returns, dtype=torch.float32)
    
    def get_tensors(self):
        """Convert buffer to tensors."""
        states = torch.FloatTensor([t.state for t in self.transitions])
        actions = torch.LongTensor([t.action for t in self.transitions])
        rewards = torch.FloatTensor([t.reward for t in self.transitions])
        next_states = torch.FloatTensor([t.next_state for t in self.transitions])
        dones = torch.FloatTensor([float(t.done) for t in self.transitions])
        log_probs = torch.stack([t.log_prob for t in self.transitions])
        
        return states, actions, rewards, next_states, dones, log_probs
