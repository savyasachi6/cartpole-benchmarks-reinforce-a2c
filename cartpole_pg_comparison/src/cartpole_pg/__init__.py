"""Main package initialization."""
from cartpole_pg.agents.reinforce import REINFORCEAgent
from cartpole_pg.agents.actor_critic import ActorCriticAgent
from cartpole_pg.core.trainer import Trainer

__version__ = "1.0.0"
__all__ = ["REINFORCEAgent", "ActorCriticAgent", "Trainer"]
