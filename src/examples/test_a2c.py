from __future__ import annotations

from src.rl_grid_world.envs.grid_world import GridWorldEnv
from src.rl_grid_world.envs.vector_grid_world import VectorGridWorldEnv
from src.rl_grid_world.agents.a2c_agent import A2CAgent, A2CConfig, train_a2c_gridworld
from src.rl_grid_world.utils.generate_wall import generate_walls
import numpy as np

if __name__ == "__main__":
    h, w = 10, 10

    obstacle_mask = generate_walls(
        h=h,
        w=w,
        obstacle_ratio=0.1,
        start=(0, 0),
        goal=(h - 1, w - 1),
    )

    # Общие параметры для обычной и векторной среды
    common_env_kwargs = dict(
        h=h,
        w=w,
        n_colors=4,
        obstacle_mask=obstacle_mask,
        pos_goal=(h - 1, w - 1),
        pos_agent=np.ones((h, w)),
        see_obstacle=True,
    )

    # Векторизованная среда с 10 копиями — для обучения
    train_env = GridWorldEnv(
        **common_env_kwargs,
        render_mode=None,
        seed=111
    )

    # Обычная (одиночная) среда — для записи GIF после обучения
    eval_env = GridWorldEnv(
        **common_env_kwargs,
        render_mode=None,
        seed=111
        
    )

    cfg = A2CConfig(
        gamma=0.99,
        lr=1e-3,
        entropy_coef=0.09,
        value_coef=0.5,
        max_grad_norm=0.5,
        num_episodes=1000,
        max_steps_per_episode=h * w,
        print_every=100,
        avg_window=100,
    )

    train_a2c_gridworld(
        env=train_env,
        config=cfg,
        project="gridworld_a2c",
        run_name="a2c_lstm_10x10_vector_10envs",
        gif_path="a2c_10x10_50envs.gif",
        eval_env=eval_env,
        use_wandb=True,
    )