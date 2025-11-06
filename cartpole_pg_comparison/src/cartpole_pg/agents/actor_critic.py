import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from .base_agent import BaseAgent
from ..networks.linear_policy import LinearPolicy
from ..networks.mlp_policy import MLPPolicy
from ..networks.value_network import LinearValueNetwork, MLPValueNetwork

class ActorCriticAgent(BaseAgent):
    """Actor-Critic with TD(0) advantage."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        actor_lr: float = 0.0005,
        critic_lr: float = 0.001,
        policy_type: str = "linear",
        hidden_dims: Optional[list[int]] = None,
        entropy_coef: float = 0.01,
        grad_clip: float = 0.5
    ):
        super().__init__(state_dim, action_dim, gamma)
        
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        
        # Initialize actor (policy)
        if policy_type == "linear":
            self.actor = LinearPolicy(state_dim, action_dim).to(self.device)
            self.critic = LinearValueNetwork(state_dim).to(self.device)
        elif policy_type == "nonlinear":
            self.actor = MLPPolicy(state_dim, action_dim, hidden_dims or [128, 64]).to(self.device)
            self.critic = MLPValueNetwork(state_dim, hidden_dims or [128, 64]).to(self.device)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
            
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Store last state for TD learning
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        
    def select_action(self, state):
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor)
        
        self.last_state = state_tensor
        self.last_action = action
        self.last_log_prob = log_prob
        
        return action, log_prob.detach()
    
    def update(self, reward: float, next_state, done: bool):
        """Update actor and critic using TD(0)."""
        if self.last_state is None:
            return {"policy_loss": 0.0, "value_loss": 0.0, "td_error": 0.0}
        
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([float(done)]).to(self.device)
        
        # Compute TD error (advantage)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor) * (1 - done_tensor)
            value = self.critic(self.last_state)
            td_error = reward_tensor + self.gamma * next_value - value
        
        # Update critic
        value_pred = self.critic(self.last_state)
        target_value = reward_tensor + self.gamma * next_value
        value_loss = nn.MSELoss()(value_pred, target_value.detach())
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # Update actor
        log_prob, entropy = self.actor.evaluate_actions(
            self.last_state, 
            torch.LongTensor([self.last_action]).to(self.device)
        )
        actor_loss = -(log_prob * td_error.detach()).mean()
        entropy_loss = -entropy.mean()
        
        total_actor_loss = actor_loss + self.entropy_coef * entropy_loss
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        
        return {
            "policy_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "td_error": td_error.item(),
            "entropy": entropy.item()
        }
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
