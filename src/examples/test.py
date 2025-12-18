import numpy as np
from rl_grid_world.envs.grid_world import GridWorldEnv

H, W = 10, 10
N_COLORS = 3 

probs = np.zeros((H, W))
probs[5, 5] = 10.0
probs += 0 

obstacles = np.zeros((H, W))
obstacles[2:5, 2] = 1 

env = GridWorldEnv(
    h=H, w=W,
    n_colors=N_COLORS,
    obstacle_mask=obstacles,
    pos_agent=probs,
    render_mode="human"
)

env.reset(seed=666)
env.render()