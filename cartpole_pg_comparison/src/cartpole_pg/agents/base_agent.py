from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def select_action(self, state):
        """Select action given state."""
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs):
        """Update agent parameters."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent checkpoint."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent checkpoint."""
        pass
