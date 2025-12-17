from enum import IntEnum
from typing import Optional, Union, Tuple
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from gymnasium import spaces

class CellType(IntEnum):
    WALL = -1
    OBSTACLE = -2
    GOAL = -3

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 h: int = 10,
                 w: int = 10,
                 obstacle_mask: Optional[np.ndarray] = None,
                 n_colors: int = 1,
                 pos_goal: Union[Tuple[int, int], np.ndarray] = (0, 0),
                 pos_agent: Union[Tuple[int, int], np.ndarray] = (1, 1),
                 see_obstacle: bool = True,
                 render_mode: Optional[str] = None
                 ):
        
        self.h = h
        self.w = w
        self.n_colors = n_colors
        self.pos_goal = np.array(pos_goal)
        self.start_pos = pos_agent
        self.see_obstacle = see_obstacle
        self.render_mode = render_mode

        if obstacle_mask is None:
            self.obstacle_mask = np.zeros((h, w), dtype=bool)
        else:
            self.obstacle_mask = np.array(obstacle_mask, dtype=bool)
            if self.obstacle_mask.shape == (w, h) and w != h:
                self.obstacle_mask = self.obstacle_mask.T
        
        self.feature_wall = n_colors
        self.feature_obstacle = n_colors + 1
        self.feature_goal = n_colors + 2
            
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.feature_goal + 1,), dtype=np.float32)
        
        self.grid = np.zeros((h + 2, w + 2), dtype=int)
        
        self.agent_pos = np.array([0, 0])
        self.see_pos = np.array([0, 0])
        
    def _init_grid(self, seed: Optional[int]):
        rng = np.random.default_rng(seed)
        
        self.grid.fill(self.feature_wall)
        
        random_floors = rng.integers(0, self.n_colors, size=(self.h, self.w))
        
        inner_area = np.where(self.obstacle_mask,
                              self.feature_obstacle,
                              random_floors)
        
        self.grid[1:-1, 1:-1] = inner_area
        
        self.grid[tuple(self.pos_goal + 1)] = self.feature_goal

    def _get_obs(self):
        obs_vec = np.zeros((self.feature_goal + 1,), dtype=np.float32)
        val = self.grid[tuple(self.see_pos)]
        obs_vec[val] = 1.0
        return obs_vec

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_grid(seed)
        
        pos_config = np.array(self.start_pos)
        
        if pos_config.ndim == 1:
            self.agent_pos = pos_config
        else:
            probs = np.array(pos_config, dtype=np.float32)
            
            if probs.shape == (self.w, self.h) and self.w != self.h:
                 probs = probs.T
                 
            probs[self.obstacle_mask] = 0
            probs[tuple(self.pos_goal)] = 0
            
            total = probs.sum()
            if total > 0:
                probs /= total
                flat_idx = self.np_random.choice(probs.size, p=probs.flatten())
                coords = np.unravel_index(flat_idx, probs.shape)
                self.agent_pos = np.array(coords)
            else:
                self.agent_pos = np.array([1, 1])

        self.see_pos = self.agent_pos.copy()
        
        return self._get_obs(), {}

    def step(self, action: int):
        v = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        
        move = v[action]
        target_pos = self.agent_pos + move
        
        see_value = self.grid[tuple(target_pos)]
        
        reward = 0.0
        terminated = False
        
        is_blocked = (see_value == self.feature_wall) or (see_value == self.feature_obstacle)
        
        if is_blocked:
            if self.see_obstacle:
                self.see_pos = target_pos
            else:
                self.see_pos = self.agent_pos
        else:
            self.agent_pos = target_pos
            self.see_pos = target_pos 
            
            if see_value == self.feature_goal:
                reward = 1.0
                terminated = True
                
        return self._get_obs(), reward, terminated, False, {}
    
    def render(self):
        if self.render_mode == "human":
            self.render_map()

    def render_map(self):
        if plt.get_fignums():
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(9, 7))
        
        ax = plt.gca()

        base_cmap = plt.get_cmap('Set3')
        
        palette_colors = []
        for i in range(self.n_colors):
            palette_colors.append(base_cmap(i % 12))
            
        palette_colors.append("black")       # Wall
        palette_colors.append("firebrick")   # Obstacle
        palette_colors.append("green")   # Goal
        
        my_cmap = colors.ListedColormap(palette_colors)
        
        max_val = self.feature_goal
        bounds = np.arange(-0.5, max_val + 1.5, 1)
        norm = colors.BoundaryNorm(bounds, my_cmap.N)


        ax.imshow(self.grid, cmap=my_cmap, norm=norm, origin='upper')

        ay, ax_coord = self.agent_pos
        ax.scatter(ax_coord, ay, c='gold', s=400, marker='*', label='Agent', edgecolors='black', zorder=10)

        ax.set_xticks(np.arange(-0.5, self.w + 2, 1))
        ax.set_yticks(np.arange(-0.5, self.h + 2, 1))
        ax.grid(which='major', color='gray', linestyle='-', linewidth=2)
        
        ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        patches = [mpatches.Patch(color='gold', label='Agent')]
        
        for i in range(self.n_colors):
            patches.append(mpatches.Patch(color=palette_colors[i], label=f'Floor {i}'))
            
        patches.append(mpatches.Patch(color='black', label='Wall'))
        patches.append(mpatches.Patch(color='firebrick', label='Obstacle'))
        patches.append(mpatches.Patch(color='limegreen', label='Goal'))

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"GridWorld {self.h}x{self.w}")
        plt.tight_layout()
        plt.show()