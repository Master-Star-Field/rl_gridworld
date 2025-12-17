import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """
    Класс реализации среды Grid World


    """
    def __init__(self, h=10, w=10, obstacle_mask=None):
        super().__init__()
        self.h = h
        self.w = w

        self.observation_space = spaces.Box(0, 1, shape=(...)) 
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        """
        Сброс среды
        """
        super().reset(seed=seed)
        return 

    def step(self, action):
        return 