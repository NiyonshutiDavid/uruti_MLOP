import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json
import os
import glob
from typing import Dict, List, Any

class RLResultsAnalyzer:
    def __init__(self, runs_dir: str = "runs", output_dir: str = "reports/plots"):
        self.runs_dir = runs_dir
        self.output_dir = output_dir
        self.set_plot_style()
        os.makedirs(output_dir, exist_ok=True)
        
    def set_plot_style(self):
        """Set consistent plotting style for publication"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl", 4)
        
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 18
        
    def load_monitor_data(self, monitor_file: str) -> Dict[str, Any]:
        """Load data from monitor.csv file"""
        try:
            # Read monitor CSV, skipping the first row which contains metadata
            df = pd.read_csv(monitor_file, skiprows=1)
            data = {}
            
            if 'r' in df.columns:
                data['rewards'] = df['r'].values
            if 'l' in df.columns:
                data['episode_lengths'] = df['l'].values
            if 't' in df.columns:
                data['timesteps'] = df['t'].values
            else:
                data['timesteps'] = np.arange(len(df))
                
            return data
        except Exception as e:
            print(f"Error loading {monitor_file}: {e}")
            return {}
    
    def load_run_data(self, run_path: str) -> Dict[str, Any]:
        """Load data from a single training run"""
        data = {}
        
        # Load monitor.csv if exists
        monitor_file = os.path.join(run_path, "monitor.csv")
        if os.path.exists(monitor_file):
            monitor_data = self.load_monitor_data(monitor_file)
            if monitor_data:
                data['monitor'] = monitor_data
        
        # Load training metrics
        metrics_file = os.path.join(run_path, "training_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data['metrics'] = json.load(f)
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")
        
        # Load config
        config_file = os.path.join(run_path, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data['config'] = json.load(f)
            except Exception as e:
                print(f"Error loading {config_file}: {e}")
        
        return data
    
    def get_algorithm_runs(self, algorithm: str) -> List[Dict[str, Any]]:
        """Get all runs for a specific algorithm"""
        pattern = os.path.join(self.runs_dir, f"{algorithm}_*")
        run_dirs = glob.glob(pattern)
        
        runs_data = []
        for run_dir in run_dirs:
            if os.path.isdir(run_dir):
                run_data = self.load_run_data(run_dir)
                if run_data and 'monitor' in run_data:  # Only add if we have monitor data
                    run_data['path'] = run_dir
                    run_data['algorithm'] = algorithm
                    runs_data.append(run_data)
        
        print(f"Found {len(runs_data)} runs for {algorithm}")
        return runs_data
    
    def plot_cumulative_rewards_comparison(self):
        """Plot cumulative rewards for best runs of each algorithm"""
        algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        best_runs = {}
        
        for idx, algorithm in enumerate(algorithms):
            runs = self.get_algorithm_runs(algorithm)
            if not runs:
                print(f"No runs found for {algorithm}")
                continue
                
            # Find best run based on final reward
            best_run = None
            best_final_reward = -float('inf')
            
            for run in runs:
                if 'monitor' in run and len(run['monitor']['rewards']) > 0:
                    final_reward = run['monitor']['rewards'][-1]
                    if final_reward > best_final_reward:
                        best_final_reward = final_reward
                        best_run = run
            
            if best_run:
                rewards = best_run['monitor']['rewards']
                episodes = np.arange(len(rewards))
                
                # Smooth rewards for better visualization
                window = min(50, len(rewards) // 10)
                if window > 1:
                    smoothed = pd.Series(rewards).rolling(window=window, center=True).mean().values
                else:
                    smoothed = rewards
                
                axes[idx].plot(episodes, rewards, alpha=0.3, color='gray', label='Raw')
                axes[idx].plot(episodes, smoothed, linewidth=2, label='Smoothed')
                axes[idx].axhline(y=best_final_reward, color='red', linestyle='--', 
                                alpha=0.7, label=f'Final: {best_final_reward:.2f}')
                
                axes[idx].set_title(f'{algorithm.upper()} - Best Run', fontweight='bold')
                axes[idx].set_xlabel('Episodes')
                axes[idx].set_ylabel('Cumulative Reward')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
                
                best_runs[algorithm] = {
                    'rewards': rewards,
                    'smoothed': smoothed,
                    'final_reward': best_final_reward,
                    'config': best_run.get('config', {})
                }
            else:
                print(f"No valid best run found for {algorithm}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cumulative_rewards_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return best_runs
    
    def plot_training_stability(self):
        """Plot training stability metrics"""
        algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        stability_data = {}
        
        for idx, algorithm in enumerate(algorithms):
            row = idx // 2
            col = idx % 2
            
            runs = self.get_algorithm_runs(algorithm)
            if not runs:
                continue
                
            stability_scores = []
            final_rewards = []
            configs = []
            
            for run in runs:
                if 'monitor' in run and len(run['monitor']['rewards']) > 10:
                    rewards = run['monitor']['rewards']
                    # Use last 20% of episodes for stability calculation
                    stable_start = len(rewards) * 4 // 5
                    stable_rewards = rewards[stable_start:]
                    
                    if len(stable_rewards) > 0:
                        mean_reward = np.mean(stable_rewards)
                        std_reward = np.std(stable_rewards)
                        stability_score = mean_reward / (std_reward + 1e-8)  # Avoid division by zero
                        
                        stability_scores.append(stability_score)
                        final_rewards.append(rewards[-1])
                        configs.append(run.get('config', {}))
            
            if stability_scores:
                # Plot stability vs performance
                scatter = axes[row, col].scatter(stability_scores, final_rewards, 
                                               alpha=0.7, s=60, c=range(len(stability_scores)), 
                                               cmap='viridis')
                axes[row, col].set_title(f'{algorithm.upper()} - Stability Analysis', fontweight='bold')
                axes[row, col].set_xlabel('Stability Score (Mean/Std)')
                axes[row, col].set_ylabel('Final Reward')
                axes[row, col].grid(True, alpha=0.3)
                
                # Add colorbar to show run progression
                plt.colorbar(scatter, ax=axes[row, col], label='Run Index')
                
                stability_data[algorithm] = {
                    'stability_scores': stability_scores,
                    'final_rewards': final_rewards,
                    'configs': configs
                }
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_stability.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return stability_data
    
    def plot_convergence_speed(self):
        """Plot episodes to convergence for each algorithm"""
        algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
        
        convergence_data = {}
        
        for algorithm in algorithms:
            runs = self.get_algorithm_runs(algorithm)
            convergence_episodes = []
            
            for run in runs:
                if 'monitor' in run and len(run['monitor']['rewards']) > 20:
                    rewards = run['monitor']['rewards']
                    max_reward = np.max(rewards)
                    target_reward = 0.8 * max_reward  # 80% of max reward
                    
                    # Find first episode where reward reaches target and stays
                    for i in range(len(rewards) - 10):
                        window = rewards[i:i+10]
                        if np.mean(window) >= target_reward and np.all(window >= 0.7 * target_reward):
                            convergence_episodes.append(i)
                            break
                    else:
                        convergence_episodes.append(len(rewards))
            
            if convergence_episodes:
                convergence_data[algorithm] = convergence_episodes
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        positions = []
        box_data = []
        labels = []
        
        for idx, algorithm in enumerate(algorithms):
            if algorithm in convergence_data:
                positions.append(idx + 1)
                box_data.append(convergence_data[algorithm])
                labels.append(algorithm.upper())
        
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.set_title('Convergence Speed Comparison', fontweight='bold', pad=20)
            ax.set_ylabel('Episodes to Convergence')
            ax.set_xlabel('Algorithm')
            ax.grid(True, alpha=0.3)
            
            # Add mean values on the plot
            for i, data in enumerate(box_data):
                mean_val = np.mean(data)
                ax.text(positions[i], mean_val, f'{mean_val:.0f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'convergence_speed.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return convergence_data
    
    def plot_hyperparameter_analysis(self):
        """Analyze hyperparameter sensitivity"""
        algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
        
        for algorithm in algorithms:
            runs = self.get_algorithm_runs(algorithm)
            if len(runs) < 3:  # Need multiple runs for analysis
                print(f"Not enough runs for {algorithm} hyperparameter analysis: {len(runs)}")
                continue
                
            # Extract hyperparameters and performance
            hp_data = []
            for run in runs:
                config = run.get('config', {})
                if algorithm in config:
                    alg_config = config[algorithm]
                else:
                    alg_config = config
                
                # FIXED: Check if rewards array exists and has elements
                performance = 0
                if 'monitor' in run and 'rewards' in run['monitor'] and len(run['monitor']['rewards']) > 0:
                    performance = run['monitor']['rewards'][-1]
                
                hp_entry = {'performance': performance}
                
                # Add common hyperparameters
                common_params = ['learning_rate', 'gamma', 'batch_size', 'ent_coef']
                for param in common_params:
                    if param in alg_config:
                        hp_entry[param] = alg_config[param]
                
                # Add algorithm-specific parameters
                if algorithm == 'dqn':
                    dqn_params = ['buffer_size', 'exploration_final_eps', 'exploration_fraction']
                    for param in dqn_params:
                        if param in alg_config:
                            hp_entry[param] = alg_config[param]
                elif algorithm == 'ppo':
                    ppo_params = ['n_steps', 'ent_coef']
                    for param in ppo_params:
                        if param in alg_config:
                            hp_entry[param] = alg_config[param]
                
                hp_data.append(hp_entry)
            
            df = pd.DataFrame(hp_data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 2:  # Need at least performance + one hyperparameter
                # Create correlation heatmap
                corr_matrix = df[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                           ax=ax, square=True, fmt='.2f')
                ax.set_title(f'{algorithm.upper()} - Hyperparameter Correlation', fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{algorithm}_hyperparameter_correlation.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create scatter plots for top correlated parameters
                performance_corr = corr_matrix['performance'].abs().sort_values(ascending=False)
                top_params = performance_corr[1:4].index.tolist()  # Top 3 excluding performance itself
                
                if top_params:
                    fig, axes = plt.subplots(1, min(3, len(top_params)), figsize=(15, 5))
                    if len(top_params) == 1:
                        axes = [axes]
                    
                    for i, param in enumerate(top_params[:3]):
                        axes[i].scatter(df[param], df['performance'], alpha=0.6)
                        axes[i].set_xlabel(param)
                        axes[i].set_ylabel('Final Performance')
                        axes[i].set_title(f'{param} vs Performance')
                        axes[i].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'{algorithm}_hyperparameter_scatter.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
    
    def generate_performance_summary(self):
        """Generate a comprehensive performance summary"""
        algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
        
        summary_data = []
        
        for algorithm in algorithms:
            runs = self.get_algorithm_runs(algorithm)
            if not runs:
                continue
                
            final_rewards = []
            convergence_episodes = []
            stability_scores = []
            
            for run in runs:
                if 'monitor' in run and len(run['monitor']['rewards']) > 0:
                    rewards = run['monitor']['rewards']
                    final_rewards.append(rewards[-1])
                    
                    # Calculate convergence
                    max_reward = np.max(rewards)
                    target = 0.8 * max_reward
                    for i in range(len(rewards) - 10):
                        if np.mean(rewards[i:i+10]) >= target:
                            convergence_episodes.append(i)
                            break
                    else:
                        convergence_episodes.append(len(rewards))
                    
                    # Calculate stability (last 20% of episodes)
                    stable_start = len(rewards) * 4 // 5
                    stable_rewards = rewards[stable_start:]
                    if len(stable_rewards) > 0:
                        mean_stable = np.mean(stable_rewards)
                        std_stable = np.std(stable_rewards)
                        stability_scores.append(mean_stable / (std_stable + 1e-8))
            
            if final_rewards:
                summary_data.append({
                    'Algorithm': algorithm.upper(),
                    'Best Final Reward': np.max(final_rewards),
                    'Mean Final Reward': np.mean(final_rewards),
                    'Std Final Reward': np.std(final_rewards),
                    'Mean Convergence Episodes': np.mean(convergence_episodes),
                    'Mean Stability Score': np.mean(stability_scores) if stability_scores else 0,
                    'Number of Runs': len(runs)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        if not summary_df.empty:
            summary_df.to_csv(os.path.join(self.output_dir, 'performance_summary.csv'), index=False)
            
            # Create summary plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Best Final Reward
            axes[0,0].bar(summary_df['Algorithm'], summary_df['Best Final Reward'], 
                         color=plt.cm.Set3(np.linspace(0, 1, len(summary_df))))
            axes[0,0].set_title('Best Final Reward by Algorithm')
            axes[0,0].set_ylabel('Reward')
            
            # Mean Convergence Episodes
            axes[0,1].bar(summary_df['Algorithm'], summary_df['Mean Convergence Episodes'],
                         color=plt.cm.Set3(np.linspace(0, 1, len(summary_df))))
            axes[0,1].set_title('Mean Episodes to Convergence')
            axes[0,1].set_ylabel('Episodes')
            
            # Mean Stability Score
            axes[1,0].bar(summary_df['Algorithm'], summary_df['Mean Stability Score'],
                         color=plt.cm.Set3(np.linspace(0, 1, len(summary_df))))
            axes[1,0].set_title('Mean Stability Score')
            axes[1,0].set_ylabel('Stability (Mean/Std)')
            
            # Number of Runs
            axes[1,1].bar(summary_df['Algorithm'], summary_df['Number of Runs'],
                         color=plt.cm.Set3(np.linspace(0, 1, len(summary_df))))
            axes[1,1].set_title('Number of Training Runs')
            axes[1,1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance_summary.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return summary_df
    
    def generate_all_plots(self):
        """Generate all plots for the report"""
        print("Generating RL performance analysis plots...")
        
        # Create all plots
        best_runs = self.plot_cumulative_rewards_comparison()
        stability_data = self.plot_training_stability()
        convergence_data = self.plot_convergence_speed()
        
        # Skip hyperparameter analysis if it fails
        try:
            self.plot_hyperparameter_analysis()
        except Exception as e:
            print(f"Hyperparameter analysis failed: {e}")
            print("Continuing with other plots...")
        
        summary_df = self.generate_performance_summary()
        
        print(f"All plots saved to: {self.output_dir}")
        
        if not summary_df.empty:
            print("\nPerformance Summary:")
            print(summary_df.to_string(index=False))
        else:
            print("\nNo performance data found!")
        
        return {
            'best_runs': best_runs,
            'stability_data': stability_data,
            'convergence_data': convergence_data,
            'summary': summary_df
        }

# Usage example
if __name__ == "__main__":
    analyzer = RLResultsAnalyzer()
    results = analyzer.generate_all_plots()