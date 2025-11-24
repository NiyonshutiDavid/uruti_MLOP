import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D

class PitchVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def create_3d_avatar(self, confidence, engagement, clarity):
        """Create a 3D visualization of the founder's performance"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a simple 3D avatar representation
        # Head
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = 0.5 * np.outer(np.cos(u), np.sin(v))
        y = 0.5 * np.outer(np.sin(u), np.sin(v))
        z = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v)) + 2
        
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.7)
        
        # Body (simplified)
        body_x = [-0.3, 0.3, 0.3, -0.3]
        body_y = [-0.2, -0.2, 0.2, 0.2]
        body_z = [1, 1, 0, 0]
        
        # Arms based on engagement
        arm_extension = engagement * 0.4
        ax.plot([-0.3, -0.3 - arm_extension], [0, 0], [1.5, 1.2], 'b-', linewidth=3)
        ax.plot([0.3, 0.3 + arm_extension], [0, 0], [1.5, 1.2], 'b-', linewidth=3)
        
        # Posture based on confidence
        posture_angle = confidence * 0.3
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 3)
        
        ax.set_title(f'Pitch Performance\nConfidence: {confidence:.2f}, Engagement: {engagement:.2f}, Clarity: {clarity:.2f}')
        
        return fig, ax
    
    def plot_training_metrics(self, metrics_data, save_path=None):
        """Plot training metrics over time"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot cumulative rewards
        ax1.plot(metrics_data['timesteps'], metrics_data['episode_rewards'])
        ax1.set_title('Cumulative Rewards')
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Reward')
        
        # Plot episode lengths
        ax2.plot(metrics_data['timesteps'], metrics_data['episode_lengths'])
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Steps')
        
        # Create combined metrics plot
        episodes = range(len(metrics_data['episode_rewards']))
        ax3.plot(episodes, metrics_data['episode_rewards'], label='Rewards')
        ax3.plot(episodes, [l/10 for l in metrics_data['episode_lengths']], label='Lengths/10')
        ax3.set_title('Training Progress')
        ax3.set_xlabel('Episodes')
        ax3.legend()
        
        # Learning curve
        window = 10
        if len(metrics_data['episode_rewards']) >= window:
            moving_avg = np.convolve(metrics_data['episode_rewards'], 
                                   np.ones(window)/window, mode='valid')
            ax4.plot(range(len(moving_avg)), moving_avg)
            ax4.set_title(f'Moving Average (window={window})')
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Average Reward')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training metrics plot saved as: {save_path}")
        
        return fig