import gymnasium as gym
import numpy as np
import pygame
import cv2
from gymnasium import spaces
import random

class PitchCoachEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super(PitchCoachEnv, self).__init__()
        
        # Observation space: [confidence, engagement, clarity, pace, slide_progress, time_remaining]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),  # Minimum values
            high=np.array([1, 1, 1, 2, 1, 1]), # Maximum values
            dtype=np.float32
        )
        
        # Action space: 6 discrete actions
        # 0: Maintain current style
        # 1: Increase energy/enthusiasm
        # 2: Use gestures
        # 3: Make eye contact
        # 4: Change slide
        # 5: Use storytelling
        self.action_space = spaces.Discrete(6)
        
        # State variables
        self.confidence = 0.5
        self.engagement = 0.5
        self.clarity = 0.5
        self.pace = 1.0  # Normal pace
        self.slide_progress = 0.0
        self.time_remaining = 1.0
        self.current_slide = 0
        self.total_slides = 10
        
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.window_size = 800
        
        # 3D visualization components (simplified with 2D for now)
        self.founder_sprite = None
        self.audience_sprites = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state
        self.confidence = 0.5 + random.uniform(-0.1, 0.1)
        self.engagement = 0.5 + random.uniform(-0.1, 0.1)
        self.clarity = 0.5 + random.uniform(-0.1, 0.1)
        self.pace = 1.0
        self.slide_progress = 0.0
        self.time_remaining = 1.0
        self.current_slide = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def _get_obs(self):
        return np.array([
            self.confidence,
            self.engagement,
            self.clarity,
            self.pace,
            self.slide_progress,
            self.time_remaining
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            'current_metrics': {
                'confidence': self.confidence,
                'engagement': self.engagement,
                'clarity': self.clarity,
                'pace': self.pace
            },
            'current_slide': self.current_slide,
            'slides_remaining': self.total_slides - self.current_slide
        }
    
    def step(self, action):
        # Update state based on action
        reward = 0
        
        # Action effects
        if action == 0:  # Maintain
            reward += 0.1
        elif action == 1:  # Increase energy
            self.confidence = min(1.0, self.confidence + 0.1)
            self.engagement = min(1.0, self.engagement + 0.15)
            reward += 0.3
        elif action == 2:  # Use gestures
            self.engagement = min(1.0, self.engagement + 0.1)
            self.clarity = min(1.0, self.clarity + 0.05)
            reward += 0.2
        elif action == 3:  # Make eye contact
            self.engagement = min(1.0, self.engagement + 0.2)
            reward += 0.4
        elif action == 4:  # Change slide
            if self.current_slide < self.total_slides - 1:
                self.current_slide += 1
                self.slide_progress = self.current_slide / self.total_slides
                reward += 0.5
                # Slide change affects metrics
                self.clarity = max(0, self.clarity - 0.05)  # Slight disruption
            else:
                reward -= 0.2  # Penalty for trying to advance beyond last slide
        elif action == 5:  # Use storytelling
            self.engagement = min(1.0, self.engagement + 0.25)
            self.confidence = min(1.0, self.confidence + 0.1)
            reward += 0.6
        
        # Natural decay and dynamics
        self.confidence = max(0, self.confidence - 0.02)
        self.engagement = max(0, self.engagement - 0.03)
        self.time_remaining = max(0, self.time_remaining - 0.02)
        
        # Calculate additional rewards based on state
        state_reward = (self.confidence * 0.3 + 
                       self.engagement * 0.4 + 
                       self.clarity * 0.3)
        reward += state_reward * 0.1
        
        # Check termination conditions
        terminated = False
        if self.time_remaining <= 0:
            terminated = True
            reward += self.slide_progress * 10  # Bonus for completing slides
        if self.current_slide >= self.total_slides - 1 and self.slide_progress >= 0.95:
            terminated = True
            reward += 10  # Completion bonus
        
        truncated = False  # We don't use truncation in this environment
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Pitch Coach - Founder Training Environment")
            self.clock = pygame.Clock()
        
        if self.clock is not None:
            self.clock.tick(self.metadata["render_fps"])
        
        # Create visualization
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        # Draw founder (simplified as a circle)
        founder_color = (0, 0, 255)  # Blue
        founder_pos = (self.window_size // 2, self.window_size // 2)
        pygame.draw.circle(canvas, founder_color, founder_pos, 50)
        
        # Draw audience (multiple circles)
        audience_color = (100, 100, 100)
        for i in range(5):
            for j in range(3):
                pos = (100 + i * 150, 100 + j * 100)
                # Color intensity based on engagement
                intensity = int(255 * self.engagement)
                color = (min(255, audience_color[0] + intensity // 3), 
                        min(255, audience_color[1] + intensity // 3),
                        min(255, audience_color[2] + intensity // 3))
                pygame.draw.circle(canvas, color, pos, 30)
        
        # Draw metrics bars
        self._draw_metric_bar(canvas, "Confidence", self.confidence, 50, 500, (0, 255, 0))
        self._draw_metric_bar(canvas, "Engagement", self.engagement, 50, 550, (255, 0, 0))
        self._draw_metric_bar(canvas, "Clarity", self.clarity, 50, 600, (0, 0, 255))
        
        # Draw progress
        self._draw_progress_bar(canvas, "Slide Progress", self.slide_progress, 50, 650)
        self._draw_progress_bar(canvas, "Time Remaining", self.time_remaining, 50, 700)
        
        if self.render_mode == "human":
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _draw_metric_bar(self, canvas, label, value, x, y, color):
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(canvas, (200, 200, 200), (x, y, bar_width, bar_height))
        pygame.draw.rect(canvas, color, (x, y, int(bar_width * value), bar_height))
        
        font = pygame.font.Font(None, 24)
        text = font.render(f"{label}: {value:.2f}", True, (0, 0, 0))
        canvas.blit(text, (x + bar_width + 10, y))
    
    def _draw_progress_bar(self, canvas, label, value, x, y):
        bar_width = 200
        bar_height = 15
        pygame.draw.rect(canvas, (200, 200, 200), (x, y, bar_width, bar_height))
        pygame.draw.rect(canvas, (0, 100, 200), (x, y, int(bar_width * value), bar_height))
        
        font = pygame.font.Font(None, 20)
        text = font.render(f"{label}: {int(value * 100)}%", True, (0, 0, 0))
        canvas.blit(text, (x + bar_width + 10, y))
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()