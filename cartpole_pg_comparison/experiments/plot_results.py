import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import uniform_filter1d

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def smooth_curve(values, window=11):
    """Smooth curve using moving average."""
    if len(values) < window:
        return values
    return uniform_filter1d(values, size=window, mode='nearest')

def load_experiment_results(results_dir: Path, exp_name: str, seeds: list):
    """Load results from multiple seeds."""
    all_steps = []
    all_returns = []
    
    for seed in seeds:
        file_path = results_dir / f"{exp_name}_seed{seed}.pkl"
        with open(file_path, "rb") as f:
            metrics = pickle.load(f)
        
        all_steps.append(metrics["steps"])
        all_returns.append(metrics["eval_returns"])
    
    return all_steps, all_returns

def plot_learning_curves(results_dir: Path, seeds: list, save_path: Path):
    """Plot learning curves for all methods."""
    experiments = [
        ("reinforce_linear_no_baseline", "REINFORCE Linear (No Baseline)", "blue"),
        ("reinforce_linear_baseline", "REINFORCE Linear (Baseline)", "cyan"),
        ("reinforce_nonlinear_no_baseline", "REINFORCE Nonlinear (No Baseline)", "orange"),
        ("reinforce_nonlinear_baseline", "REINFORCE Nonlinear (Baseline)", "red"),
        ("actor_critic_linear", "A2C Linear", "green"),
        ("actor_critic_nonlinear", "A2C Nonlinear", "purple")
    ]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for exp_name, label, color in experiments:
        all_steps, all_returns = load_experiment_results(results_dir, exp_name, seeds)
        
        # Align to common step grid
        max_len = min(len(s) for s in all_steps)
        steps = all_steps[0][:max_len]
        returns_array = np.array([r[:max_len] for r in all_returns])
        
        # Compute mean and std
        mean_returns = returns_array.mean(axis=0)
        std_returns = returns_array.std(axis=0)
        
        # Smooth curves
        smooth_mean = smooth_curve(mean_returns, window=5)
        smooth_std = smooth_curve(std_returns, window=5)
        
        # Plot
        ax.plot(steps, smooth_mean, label=label, color=color, linewidth=2)
        ax.fill_between(steps, smooth_mean - smooth_std, smooth_mean + smooth_std, 
                        alpha=0.2, color=color)
    
    ax.axhline(y=475, color='black', linestyle='--', linewidth=1.5, label='Solved Threshold')
    ax.set_xlabel('Environment Steps', fontsize=14)
    ax.set_ylabel('Average Return', fontsize=14)
    ax.set_title('Learning Curves: Policy Gradient Methods on CartPole-v1', fontsize=16)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def compute_steps_to_threshold(all_returns, all_steps, threshold=475):
    """Compute steps needed to exceed threshold."""
    steps_list = []
    
    for returns, steps in zip(all_returns, all_steps):
        for i, ret in enumerate(returns):
            if ret >= threshold:
                steps_list.append(steps[i])
                break
        else:
            steps_list.append(None)  # Never reached
    
    valid_steps = [s for s in steps_list if s is not None]
    
    if len(valid_steps) == 0:
        return None, None
    
    return np.mean(valid_steps), np.std(valid_steps)

def create_comparison_tables(results_dir: Path, seeds: list, save_path: Path):
    """Create comparison tables."""
    # REINFORCE comparison
    reinforce_experiments = [
        ("reinforce_linear_no_baseline", "Linear", "None"),
        ("reinforce_linear_baseline", "Linear", "Value Baseline"),
        ("reinforce_nonlinear_no_baseline", "Nonlinear", "None"),
        ("reinforce_nonlinear_baseline", "Nonlinear", "Value Baseline")
    ]
    
    print("\nREINFORCE Advantage Variants:")
    print("="*80)
    print(f"{'Policy':<15} {'Advantage':<20} {'Steps-to-475':<25} {'Best Avg Return':<20} {'Return Std'}")
    print("-"*80)
    
    for exp_name, policy, advantage in reinforce_experiments:
        all_steps, all_returns = load_experiment_results(results_dir, exp_name, seeds)
        
        steps_mean, steps_std = compute_steps_to_threshold(all_returns, all_steps)
        
        final_returns = [r[-1] for r in all_returns]
        best_avg = np.mean(final_returns)
        return_std = np.std(final_returns)
        
        if steps_mean is not None:
            steps_str = f"{steps_mean:.0f} ± {steps_std:.0f}"
        else:
            steps_str = "Not reached"
        
        print(f"{policy:<15} {advantage:<20} {steps_str:<25} {best_avg:>8.1f} ± {return_std:<6.1f}   {return_std:>6.1f}")
    
    # Actor-Critic comparison
    ac_experiments = [
        ("actor_critic_linear", "A2C Linear", "TD(0)"),
        ("actor_critic_nonlinear", "A2C Nonlinear", "TD(0)")
    ]
    
    print("\n\nActor-Critic Advantage Variants:")
    print("="*80)
    print(f"{'AC Variant':<20} {'Advantage':<15} {'Steps-to-475':<25} {'Best Avg Return':<20} {'Return Std'}")
    print("-"*80)
    
    for exp_name, variant, advantage in ac_experiments:
        all_steps, all_returns = load_experiment_results(results_dir, exp_name, seeds)
        
        steps_mean, steps_std = compute_steps_to_threshold(all_returns, all_steps)
        
        final_returns = [r[-1] for r in all_returns]
        best_avg = np.mean(final_returns)
        return_std = np.std(final_returns)
        
        if steps_mean is not None:
            steps_str = f"{steps_mean:.0f} ± {steps_std:.0f}"
        else:
            steps_str = "Not reached"
        
        print(f"{variant:<20} {advantage:<15} {steps_str:<25} {best_avg:>8.1f} ± {return_std:<6.1f}   {return_std:>6.1f}")

def main():
    """Generate all plots and tables."""
    results_dir = Path("results/experiments")
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123, 456, 789, 1024]
    
    # Plot learning curves
    print("Generating learning curves...")
    plot_learning_curves(results_dir, seeds, figures_dir)
    
    # Create comparison tables
    create_comparison_tables(results_dir, seeds, figures_dir)
    
    print(f"\nAll plots saved to {figures_dir}")

if __name__ == "__main__":
    main()
