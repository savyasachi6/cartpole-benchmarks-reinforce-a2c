# CartPole Policy Gradient Comparison

Professional implementation comparing REINFORCE and Actor-Critic methods on CartPole-v1.

## Installation
Clone repository
cd cartpole_pg_comparison

Install package
pip install -e 

## Usage

### Run all experiments (5 seeds each):


## Project Structure
cartpole_pg_comparison/
├── src/cartpole_pg/ # Core package
│ ├── agents/ # Agent implementations
│ ├── networks/ # Neural network architectures
│ ├── utils/ # Utilities and buffers
│ └── core/ # Training and evaluation
├── experiments/ # Experiment scripts
├── config/ # Configuration files
└── results/ # Output directory

## Methods Implemented

1. **REINFORCE Linear (No Baseline)** - Linear policy with raw returns
2. **REINFORCE Linear (Baseline)** - Linear policy with value baseline
3. **REINFORCE Nonlinear (No Baseline)** - MLP policy with raw returns
4. **REINFORCE Nonlinear (Baseline)** - MLP policy with value baseline
5. **A2C Linear** - Linear Actor-Critic with TD(0)
6. **A2C Nonlinear** - MLP Actor-Critic with TD(0)

## Configuration

Edit `config/experiment_config.yaml` to modify hyperparameters, training budget, or seeds.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy, Matplotlib, Seaborn
