# CartPole Benchmarks: REINFORCE and A2C

A modular benchmarking suite for policy gradient methods on the classic CartPole environment, featuring **REINFORCE** and **Actor-Critic (A2C)** implementations with linear and nonlinear architectures. Designed for clarity, reproducibility, and educational insight.

---

## 🚀 Features

- Modular, pip-installable Python package (`pip install -e .`)
- Clean separation of agents, networks, utilities, and training core
- Linear and deep (MLP-based) policies and critics
- REINFORCE with and without value baselines
- Actor-Critic (A2C) with TD(0) advantage for discrete control
- Multi-seed experimentation for robust statistical results
- Automated plotting, metric exports, and report generation

---

## ⚙️ Installation

```bash
git clone https://github.com/<yourusername>/cartpole-benchmarks-reinforce-a2c.git
cd cartpole-benchmarks-reinforce-a2c

python -m venv venv
source venv/bin/activate          # On Windows: .\venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

---

## 🧠 Usage

### Run All Experiments (Default: 5 Seeds)

```bash
python experiments/run_experiments.py
```

### Generate Learning Curves and Reports

```bash
python experiments/plot_results.py
```

All outputs—including figures, logs, and tables—are automatically saved in the `results/` directory.

---

## 📁 Project Structure

```
cartpole-benchmarks-reinforce-a2c/
├── setup.py                     # Package configuration
├── requirements.txt             # Dependencies
├── .gitignore
│
├── src/
│   └── cartpole_pg/
│       ├── __init__.py
│       ├── agents/              # REINFORCE, A2C agent classes
│       ├── networks/            # Linear and MLP policy/value networks
│       ├── utils/               # Buffers, logging, metrics, and helpers
│       └── core/                # Training and evaluation loops
│
├── config/
│   └── experiment_config.yaml   # Environment, seeds, and hyperparameters
│
├── experiments/
│   ├── run_experiments.py       # Master experiment script
│   └── plot_results.py          # Visualization and reporting
│
├── results/
│   ├── experiments/             # Metrics from each run and seed
│   ├── figures/                 # Learning curves and loss plots
│   ├── reports/                 # Markdown and LaTeX summaries
│   ├── checkpoints/             # Saved model weights
│   └── logs/                    # Training logs
│
├── tests/
│   ├── test_agents.py
│   └── test_networks.py
│
└── README.md
```

---

## 🧩 Methods Implemented

1. **REINFORCE Linear (No Baseline):**  
   Linear policy trained with Monte Carlo returns.

2. **REINFORCE Linear (Baseline):**  
   Linear policy augmented with a learned value baseline.

3. **REINFORCE Nonlinear (No Baseline):**  
   MLP policy trained using Monte Carlo returns.

4. **REINFORCE Nonlinear (Baseline):**  
   MLP policy and value network with advantage-based updates.

5. **A2C Linear:**  
   Linear Actor-Critic using TD(0) advantage estimates.

6. **A2C Nonlinear:**  
   MLP-based Actor-Critic employing TD(0) advantage updates.

---

## ⚙️ Configuration

All experimental parameters are specified in:

```bash
config/experiment_config.yaml
```

You may configure:
- Learning rates, entropy coefficients, and gradient clipping  
- Hidden layer dimensions for nonlinear models  
- Total environment steps and evaluation frequency  
- Random seeds for reproducibility  

---

## 🧾 Requirements

- **Python** ≥ 3.8  
- **PyTorch** ≥ 2.0.0  
- **Gymnasium** ≥ 0.29.0  
- **NumPy**, **Matplotlib**, **Seaborn**, **tqdm**, **PyYAML**

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔁 Reproducibility

To reproduce the full benchmark suite with default configurations:

```bash
python experiments/run_experiments.py
python experiments/plot_results.py
```

All generated results—including metrics, plots, and tables—are saved in the `results/` directory.

---

## 📊 Outputs

- **Learning Curves:**  
  Mean ± standard deviation across 5 seeds for each method.

- **Loss and Stability Plots:**  
  Per-update policy and value loss, return variance.

- **Comparison Tables:**  
  Summaries of sample efficiency and final performance.

- **Policy Histograms:**  
  Action probability distributions at convergence.

- **Markdown/LaTeX Reports:**  
  Automatically generated summaries located in `results/reports/`.

---

## 🧹 .gitignore

This repository uses a comprehensive `.gitignore` file to maintain a clean workspace:

```
__pycache__/
*.py[cod]
build/
dist/
.eggs/
*.egg-info/
*.egg
.env
.venv
env/
venv/
ENV/
.ipynb_checkpoints
results/
checkpoints/
logs/
figures/
*.log
*.png
*.jpg
.DS_Store
.vscode/
.idea/
```

---

## 💡 Summary

Our experiments underscore a familiar observation in reinforcement learning research: **expressive function approximators and well-chosen baselines profoundly affect training stability and performance**.  
Linear REINFORCE agents tended to plateau early, whereas nonlinear models achieved rapid improvement, particularly with value baselines. Interestingly, the Actor-Critic variants—often considered the gold standard—faltered under default hyperparameters, illustrating that theoretical guarantees alone rarely ensure practical success.  
The resulting curves narrate a story of exploration, instability, and eventual adaptation—a microcosm of learning itself.
