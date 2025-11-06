import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from .base_agent import BaseAgent
from ..networks.linear_policy import LinearPolicy
from ..networks.mlp_policy import MLPPolicy
from ..networks.value_network import LinearValueNetwork, MLPValueNetwork
from ..utils.buffer import EpisodeBuffer, Transition

class REINFORCEAgent(BaseAgent):
    """REINFORCE algorithm with optional value baseline."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 0.001,
        use_baseline: bool = False,
        policy_type: str = "linear",
        hidden_dims: Optional[list[int]] = None,
        entropy_coef: float = 0.01,
        grad_clip: float = 1.0
    ):
        super().__init__(state_dim, action_dim, gamma)
        
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        
        # Initialize policy network
        if policy_type == "linear":
            self.policy = LinearPolicy(state_dim, action_dim).to(self.device)
        elif policy_type == "nonlinear":
            self.policy = MLPPolicy(state_dim, action_dim, hidden_dims or [128, 64]).to(self.device)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
            
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize value network if using baseline
        if self.use_baseline:
            if policy_type == "linear":
                self.value_net = LinearValueNetwork(state_dim).to(self.device)
            else:
                self.value_net = MLPValueNetwork(state_dim, hidden_dims or [128, 64]).to(self.device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
            
        self.buffer = EpisodeBuffer()
        
    def select_action(self, state):
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.policy.get_action(state_tensor)
        return action, log_prob.detach()
    
    def update(self):
        """Update policy using REINFORCE with optional baseline."""
        if len(self.buffer.transitions) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Get data from buffer
        states, actions, rewards, next_states, dones, old_log_probs = self.buffer.get_tensors()
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        # Compute returns
        returns = self.buffer.compute_returns(self.gamma).to(self.device)
        
        # Compute advantages
        if self.use_baseline:
            with torch.no_grad():
                values = self.value_net(states)
            advantages = returns - values
            
            # Update value network
            value_pred = self.value_net(states)
            value_loss = nn.MSELoss()(value_pred, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
            self.value_optimizer.step()
        else:
            advantages = returns
            value_loss = torch.tensor(0.0)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss
        log_probs, entropy = self.policy.evaluate_actions(states, actions)
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -entropy.mean()
        
        total_loss = policy_loss + self.entropy_coef * entropy_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item() if self.use_baseline else 0.0,
            "entropy": entropy.mean().item(),
            "mean_return": returns.mean().item()
        }
    
    def save(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
        }
        if self.use_baseline:
            checkpoint["value_state_dict"] = self.value_net.state_dict()
            checkpoint["value_optimizer_state_dict"] = self.value_optimizer.state_dict()
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        if self.use_baseline:
            self.value_net.load_state_dict(checkpoint["value_state_dict"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
