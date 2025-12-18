import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
from src.rl_grid_world.envs.grid_world import GridWorldEnv
from src.rl_grid_world.agents.drqn_agent import DRQNLightning
from src.rl_grid_world.agents.drqn_agent import train_drqn_gridworld
from src.rl_grid_world.utils.save_gif import save_episode_gif
from src.rl_grid_world.utils.generate_wall import generate_walls
import wandb
if __name__ == "__main__":

    h, w = 10, 10
    start = (0, 0)  
    goal = (h - 1, w - 1)
    obstacle_ratio = 0.2

    obstacle_mask = generate_walls(
        h=h,
        w=w,
        obstacle_ratio=obstacle_ratio,
        start=start,
        goal=goal,
    )

    env_params = dict(
        h=h,
        w=w,
        n_colors=4,
        obstacle_mask=obstacle_mask,
        pos_goal=goal,
        pos_agent=np.ones((h, w)),
        see_obstacle=True,
        render_mode=None,
    )

    train_drqn_gridworld(
        env_params=env_params,
        max_steps=24000,
        lr=1e-3,
        gamma=0.99,
        seq_len=20,
        burn_in=5,
        batch_size=32,
        buffer_size=500,
        min_episodes=10,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=10_000,
        sync_rate=1_000,
        hidden_dim=128,
        avg_window=100,
        project="gridworld_drqn",
        run_name="drqn_dueling_double_20x20",
        gif_path="20to20c4random_pos.gif",
    )