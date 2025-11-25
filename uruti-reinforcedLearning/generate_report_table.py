import pandas as pd
import json
import glob
import os
from typing import List, Dict

class ReportTableGenerator:
    def __init__(self, configs_dir: str = "configs", runs_dir: str = "runs", output_dir: str = "reports/tables"):
        self.configs_dir = configs_dir
        self.runs_dir = runs_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_config_performance(self, algorithm: str) -> List[Dict]:
        """Load configuration and performance data for an algorithm"""
        runs_data = []
        
        # Load config files
        config_pattern = os.path.join(self.configs_dir, algorithm, "run_*.json")
        config_files = glob.glob(config_pattern)
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Extract algorithm-specific config
                if algorithm in config_data:
                    alg_config = config_data[algorithm]
                else:
                    alg_config = config_data
                
                # Find corresponding run and get performance
                run_id = os.path.basename(config_file).replace('.json', '')
                run_pattern = os.path.join(self.runs_dir, f"{algorithm}_*")
                run_dirs = glob.glob(run_pattern)
                
                performance = 0
                for run_dir in run_dirs:
                    monitor_file = os.path.join(run_dir, "monitor.csv")
                    if os.path.exists(monitor_file):
                        try:
                            df = pd.read_csv(monitor_file, skiprows=1)
                            if len(df) > 0:
                                performance = df['r'].iloc[-1] if 'r' in df.columns else 0
                                break
                        except:
                            continue
                
                run_data = {
                    'Run': run_id,
                    'Performance': performance,
                    **alg_config
                }
                runs_data.append(run_data)
                
            except Exception as e:
                print(f"Error processing {config_file}: {e}")
                continue
        
        return runs_data
    
    def generate_dqn_table(self):
        """Generate DQN results table"""
        dqn_data = self.load_config_performance('dqn')
        if dqn_data:
            df = pd.DataFrame(dqn_data)
            # Select and rename columns for report
            report_columns = {
                'Run': 'Run',
                'learning_rate': 'Learning Rate',
                'gamma': 'Gamma',
                'buffer_size': 'Replay Buffer Size',
                'batch_size': 'Batch Size',
                'exploration_final_eps': 'Final Exploration',
                'Performance': 'Mean Reward'
            }
            
            # Keep only columns that exist
            existing_cols = {k: v for k, v in report_columns.items() if k in df.columns}
            report_df = df[list(existing_cols.keys())].rename(columns=existing_cols)
            report_df = report_df.sort_values('Mean Reward', ascending=False)
            report_df.to_csv(os.path.join(self.output_dir, 'dqn_results.csv'), index=False)
            return report_df
        return None
    
    def generate_ppo_table(self):
        """Generate PPO results table"""
        ppo_data = self.load_config_performance('ppo')
        if ppo_data:
            df = pd.DataFrame(ppo_data)
            report_columns = {
                'Run': 'Run',
                'learning_rate': 'Learning Rate',
                'gamma': 'Gamma',
                'n_steps': 'N Steps',
                'batch_size': 'Batch Size',
                'ent_coef': 'Entropy Coefficient',
                'Performance': 'Mean Reward'
            }
            
            existing_cols = {k: v for k, v in report_columns.items() if k in df.columns}
            report_df = df[list(existing_cols.keys())].rename(columns=existing_cols)
            report_df = report_df.sort_values('Mean Reward', ascending=False)
            report_df.to_csv(os.path.join(self.output_dir, 'ppo_results.csv'), index=False)
            return report_df
        return None
    
    def generate_a2c_table(self):
        """Generate A2C results table"""
        a2c_data = self.load_config_performance('a2c')
        if a2c_data:
            df = pd.DataFrame(a2c_data)
            report_columns = {
                'Run': 'Run',
                'learning_rate': 'Learning Rate',
                'gamma': 'Gamma',
                'n_steps': 'N Steps',
                'ent_coef': 'Entropy Coefficient',
                'Performance': 'Mean Reward'
            }
            
            existing_cols = {k: v for k, v in report_columns.items() if k in df.columns}
            report_df = df[list(existing_cols.keys())].rename(columns=existing_cols)
            report_df = report_df.sort_values('Mean Reward', ascending=False)
            report_df.to_csv(os.path.join(self.output_dir, 'a2c_results.csv'), index=False)
            return report_df
        return None
    
    def generate_reinforce_table(self):
        """Generate REINFORCE results table"""
        reinforce_data = self.load_config_performance('reinforce')
        if reinforce_data:
            df = pd.DataFrame(reinforce_data)
            report_columns = {
                'Run': 'Run',
                'learning_rate': 'Learning Rate',
                'gamma': 'Gamma',
                'n_steps': 'N Steps',
                'batch_size': 'Batch Size',
                'ent_coef': 'Entropy Coefficient',
                'Performance': 'Mean Reward'
            }
            
            existing_cols = {k: v for k, v in report_columns.items() if k in df.columns}
            report_df = df[list(existing_cols.keys())].rename(columns=existing_cols)
            report_df = report_df.sort_values('Mean Reward', ascending=False)
            report_df.to_csv(os.path.join(self.output_dir, 'reinforce_results.csv'), index=False)
            return report_df
        return None
    
    def generate_all_tables(self):
        """Generate all report tables"""
        print("Generating report tables...")
        
        dqn_table = self.generate_dqn_table()
        ppo_table = self.generate_ppo_table()
        a2c_table = self.generate_a2c_table()
        reinforce_table = self.generate_reinforce_table()
        
        print("Tables generated in:", self.output_dir)
        
        return {
            'dqn': dqn_table,
            'ppo': ppo_table,
            'a2c': a2c_table,
            'reinforce': reinforce_table
        }

if __name__ == "__main__":
    generator = ReportTableGenerator()
    tables = generator.generate_all_tables()