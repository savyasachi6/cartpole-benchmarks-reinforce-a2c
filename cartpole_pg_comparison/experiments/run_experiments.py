import os
import yaml
import pickle
import numpy as np
from pathlib import Path
from cartpole_pg import Trainer, REINFORCEAgent, ActorCriticAgent


def run_single_experiment(agent_config: dict, seed: int, results_dir: Path):
    """Run single experiment with given configuration."""
    agent_type = agent_config["type"]
    
    # Create agent
    if agent_type == "reinforce":
        agent = REINFORCEAgent(
            state_dim=4,
            action_dim=2,
            gamma=agent_config["gamma"],
            lr=agent_config["lr"],
            use_baseline=agent_config["use_baseline"],
            policy_type=agent_config["policy_type"],
            hidden_dims=agent_config.get("hidden_dims"),
            entropy_coef=agent_config["entropy_coef"],
            grad_clip=agent_config["grad_clip"]
        )
    elif agent_type == "actor_critic":
        agent = ActorCriticAgent(
            state_dim=4,
            action_dim=2,
            gamma=agent_config["gamma"],
            actor_lr=agent_config["actor_lr"],
            critic_lr=agent_config["critic_lr"],
            policy_type=agent_config["policy_type"],
            hidden_dims=agent_config.get("hidden_dims"),
            entropy_coef=agent_config["entropy_coef"],
            grad_clip=agent_config["grad_clip"]
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Create trainer
    trainer = Trainer(
        env_name="CartPole-v1",
        max_steps=agent_config["max_steps"],
        eval_frequency=agent_config["eval_frequency"],
        eval_episodes=10,
        seed=seed
    )
    
    # Train
    if agent_type == "reinforce":
        metrics = trainer.train_reinforce(agent)
    else:
        metrics = trainer.train_actor_critic(agent)
    
    # Save results
    exp_name = f"{agent_config['name']}_seed{seed}"
    save_path = results_dir / f"{exp_name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(metrics, f)
    
    print(f"Completed: {exp_name}")
    return metrics

def main():
    """Run all experiments."""
    # Load config
    with open("config/experiment_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create results directory
    results_dir = Path("results/experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = config["training"]["seeds"]
    max_steps = config["training"]["max_steps"]
    gamma = config["training"]["gamma"]
    
    # Define experiments
    experiments = [
        # Linear REINFORCE without baseline
        {
            "name": "reinforce_linear_no_baseline",
            "type": "reinforce",
            "policy_type": "linear",
            "use_baseline": False,
            "lr": config["reinforce_linear"]["lr"],
            "entropy_coef": config["reinforce_linear"]["entropy_coef"],
            "grad_clip": config["reinforce_linear"]["grad_clip"],
            "gamma": gamma,
            "max_steps": max_steps,
            "eval_frequency": config["logging"]["eval_frequency"]
        },
        # Linear REINFORCE with baseline
        {
            "name": "reinforce_linear_baseline",
            "type": "reinforce",
            "policy_type": "linear",
            "use_baseline": True,
            "lr": config["reinforce_linear"]["lr"],
            "entropy_coef": config["reinforce_linear"]["entropy_coef"],
            "grad_clip": config["reinforce_linear"]["grad_clip"],
            "gamma": gamma,
            "max_steps": max_steps,
            "eval_frequency": config["logging"]["eval_frequency"]
        },
        # Nonlinear REINFORCE without baseline
        {
            "name": "reinforce_nonlinear_no_baseline",
            "type": "reinforce",
            "policy_type": "nonlinear",
            "use_baseline": False,
            "hidden_dims": config["reinforce_nonlinear"]["hidden_dims"],
            "lr": config["reinforce_nonlinear"]["lr"],
            "entropy_coef": config["reinforce_nonlinear"]["entropy_coef"],
            "grad_clip": config["reinforce_nonlinear"]["grad_clip"],
            "gamma": gamma,
            "max_steps": max_steps,
            "eval_frequency": config["logging"]["eval_frequency"]
        },
        # Nonlinear REINFORCE with baseline
        {
            "name": "reinforce_nonlinear_baseline",
            "type": "reinforce",
            "policy_type": "nonlinear",
            "use_baseline": True,
            "hidden_dims": config["reinforce_nonlinear"]["hidden_dims"],
            "lr": config["reinforce_nonlinear"]["lr"],
            "entropy_coef": config["reinforce_nonlinear"]["entropy_coef"],
            "grad_clip": config["reinforce_nonlinear"]["grad_clip"],
            "gamma": gamma,
            "max_steps": max_steps,
            "eval_frequency": config["logging"]["eval_frequency"]
        },
        # Linear Actor-Critic
        {
            "name": "actor_critic_linear",
            "type": "actor_critic",
            "policy_type": "linear",
            "actor_lr": config["actor_critic_linear"]["actor_lr"],
            "critic_lr": config["actor_critic_linear"]["critic_lr"],
            "entropy_coef": config["actor_critic_linear"]["entropy_coef"],
            "grad_clip": config["actor_critic_linear"]["grad_clip"],
            "gamma": gamma,
            "max_steps": max_steps,
            "eval_frequency": config["logging"]["eval_frequency"]
        },
        # Nonlinear Actor-Critic
        {
            "name": "actor_critic_nonlinear",
            "type": "actor_critic",
            "policy_type": "nonlinear",
            "hidden_dims": config["actor_critic_nonlinear"]["hidden_dims"],
            "actor_lr": config["actor_critic_nonlinear"]["actor_lr"],
            "critic_lr": config["actor_critic_nonlinear"]["critic_lr"],
            "entropy_coef": config["actor_critic_nonlinear"]["entropy_coef"],
            "grad_clip": config["actor_critic_nonlinear"]["grad_clip"],
            "gamma": gamma,
            "max_steps": max_steps,
            "eval_frequency": config["logging"]["eval_frequency"]
        }
    ]
    
    # Run all experiments
    for exp_config in experiments:
        print(f"Running experiment: {exp_config['name']}")
        
        
        for seed in seeds:
            run_single_experiment(exp_config, seed, results_dir)

if __name__ == "__main__":
    main()
