import gymnasium as gym
import numpy as np
import torch
import argparse
import json
import os
from datetime import datetime
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.pitch_env import PitchCoachEnv
from utils.visualization import PitchVisualizer

class ExperimentLogger(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(ExperimentLogger, self).__init__(verbose)
        self.save_path = save_path
        self.experiment_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'training_loss': [],
            'timesteps': []
        }
        
    def _on_step(self) -> bool:
        if 'episode' in self.locals['infos'][0]:
            episode_info = self.locals['infos'][0]['episode']
            self.experiment_data['episode_rewards'].append(episode_info['r'])
            self.experiment_data['episode_lengths'].append(episode_info['l'])
            self.experiment_data['timesteps'].append(self.num_timesteps)
            
        return True
    
    def _on_training_end(self):
        # Save experiment data
        with open(os.path.join(self.save_path, 'training_metrics.json'), 'w') as f:
            json.dump(self.experiment_data, f)

def train_dqn(env, experiment_dir, config):
    """Train DQN model with hyperparameters"""
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        exploration_fraction=config['exploration_fraction'],
        exploration_final_eps=config['exploration_final_eps'],
        tensorboard_log=experiment_dir,
        verbose=1
    )
    
    logger = ExperimentLogger(experiment_dir)
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=logger,
        tb_log_name="DQN"
    )
    
    model.save(os.path.join(experiment_dir, "dqn_model"))
    return model

def train_reinforce(env, experiment_dir, config):
    """Train REINFORCE model with hyperparameters"""
    model = PPO(  # Using PPO as REINFORCE implementation
        'MlpPolicy',
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        n_steps=config['n_steps'],
        ent_coef=config['ent_coef'],
        tensorboard_log=experiment_dir,
        verbose=1
    )
    
    logger = ExperimentLogger(experiment_dir)
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=logger,
        tb_log_name="REINFORCE"
    )
    
    model.save(os.path.join(experiment_dir, "reinforce_model"))
    return model

def train_a2c(env, experiment_dir, config):
    """Train A2C model with hyperparameters"""
    model = A2C(
        'MlpPolicy',
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        n_steps=config['n_steps'],
        ent_coef=config['ent_coef'],
        tensorboard_log=experiment_dir,
        verbose=1
    )
    
    logger = ExperimentLogger(experiment_dir)
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=logger,
        tb_log_name="A2C"
    )
    
    model.save(os.path.join(experiment_dir, "a2c_model"))
    return model

def main():
    parser = argparse.ArgumentParser(description='Train RL models for Pitch Coach')
    parser.add_argument('--algorithm', type=str, required=True, 
                       choices=['dqn', 'reinforce', 'a2c', 'all'],
                       help='RL algorithm to train')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration JSON file')
    
    args = parser.parse_args()
    
    # Load training configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{args.algorithm}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create environment
    env = PitchCoachEnv()
    env = Monitor(env, experiment_dir)
    
    print(f"Starting training with {args.algorithm} algorithm...")
    print(f"Experiment directory: {experiment_dir}")
    
    # Train selected algorithm
    if args.algorithm == 'dqn' or args.algorithm == 'all':
        print("Training DQN...")
        dqn_config = config['dqn']
        train_dqn(env, experiment_dir, dqn_config)
    
    if args.algorithm == 'reinforce' or args.algorithm == 'all':
        print("Training REINFORCE...")
        reinforce_config = config['reinforce']
        train_reinforce(env, experiment_dir, reinforce_config)
    
    if args.algorithm == 'a2c' or args.algorithm == 'all':
        print("Training A2C...")
        a2c_config = config['a2c']
        train_a2c(env, experiment_dir, a2c_config)
    
    env.close()
    print("Training completed!")

if __name__ == "__main__":
    main()