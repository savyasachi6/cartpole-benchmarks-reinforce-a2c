import gymnasium as gym
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from ..agents.reinforce import REINFORCEAgent
from ..agents.actor_critic import ActorCriticAgent
from ..utils.buffer import Transition

class Trainer:
    """Trainer for policy gradient agents."""
    
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        max_steps: int = 200000,
        eval_frequency: int = 5000,
        eval_episodes: int = 10,
        seed: int = 42
    ):
        self.env_name = env_name
        self.max_steps = max_steps
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.seed = seed
        
        # Create environments
        self.train_env = gym.make(env_name)
        self.eval_env = gym.make(env_name)
        
        # Set seeds
        self.train_env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 1000)
        np.random.seed(seed)
        
    def train_reinforce(self, agent: REINFORCEAgent) -> Dict[str, List]:
        """Train REINFORCE agent."""
        metrics = {
            "steps": [],
            "eval_returns": [],
            "train_returns": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": []
        }
        
        state, _ = self.train_env.reset()
        episode_return = 0
        total_steps = 0
        
        pbar = tqdm(total=self.max_steps, desc="Training REINFORCE")
        
        while total_steps < self.max_steps:
            # Collect episode
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated
            
            agent.buffer.add(Transition(state, action, reward, next_state, done, log_prob))
            
            episode_return += reward
            state = next_state
            total_steps += 1
            pbar.update(1)
            
            # Episode ended
            if done:
                # Update agent
                losses = agent.update()
                
                metrics["train_returns"].append(episode_return)
                metrics["policy_losses"].append(losses["policy_loss"])
                metrics["value_losses"].append(losses["value_loss"])
                metrics["entropies"].append(losses["entropy"])
                
                # Reset environment
                state, _ = self.train_env.reset()
                episode_return = 0
            
            # Periodic evaluation
            if total_steps % self.eval_frequency == 0:
                eval_return = self.evaluate(agent)
                metrics["steps"].append(total_steps)
                metrics["eval_returns"].append(eval_return)
                
                pbar.set_postfix({"eval_return": f"{eval_return:.1f}"})
        
        pbar.close()
        return metrics
    
    def train_actor_critic(self, agent: ActorCriticAgent) -> Dict[str, List]:
        """Train Actor-Critic agent."""
        metrics = {
            "steps": [],
            "eval_returns": [],
            "train_returns": [],
            "policy_losses": [],
            "value_losses": [],
            "td_errors": [],
            "entropies": []
        }
        
        state, _ = self.train_env.reset()
        episode_return = 0
        total_steps = 0
        
        pbar = tqdm(total=self.max_steps, desc="Training Actor-Critic")
        
        while total_steps < self.max_steps:
            # Select action
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated
            
            # Update agent (TD learning)
            losses = agent.update(reward, next_state, done)
            
            episode_return += reward
            state = next_state
            total_steps += 1
            pbar.update(1)
            
            metrics["policy_losses"].append(losses["policy_loss"])
            metrics["value_losses"].append(losses["value_loss"])
            metrics["td_errors"].append(losses["td_error"])
            metrics["entropies"].append(losses["entropy"])
            
            # Episode ended
            if done:
                metrics["train_returns"].append(episode_return)
                state, _ = self.train_env.reset()
                episode_return = 0
            
            # Periodic evaluation
            if total_steps % self.eval_frequency == 0:
                eval_return = self.evaluate(agent)
                metrics["steps"].append(total_steps)
                metrics["eval_returns"].append(eval_return)
                
                pbar.set_postfix({"eval_return": f"{eval_return:.1f}"})
        
        pbar.close()
        return metrics
    
    def evaluate(self, agent) -> float:
        """Evaluate agent performance."""
        returns = []
        
        for _ in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            episode_return = 0
            done = False
            
            while not done:
                action, _ = agent.select_action(state)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_return += reward
            
            returns.append(episode_return)
        
        return np.mean(returns)
