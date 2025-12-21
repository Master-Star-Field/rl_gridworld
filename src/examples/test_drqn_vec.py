from __future__ import annotations

import numpy as np

from src.rl_grid_world.envs.grid_world import GridWorldEnv
from src.rl_grid_world.envs.gym_grid_world import GymVectorGridWorldEnv
from src.rl_grid_world.agents.drqn_agent import DRQNLightning, train_drqn_gridworld
from src.rl_grid_world.utils.generate_wall import generate_walls


if __name__ == "__main__":
    h, w = 10, 10

    # Генерируем препятствия
    obstacle_mask = generate_walls(
        h=h,
        w=w,
        obstacle_ratio=0.1,
        start=(0, 0),
        goal=(h - 1, w - 1),
    )

    # Общие параметры для одиночной среды (и для фабрики векторной)
    common_env_kwargs = dict(
        h=h,
        w=w,
        n_colors=4,
        obstacle_mask=obstacle_mask,
        pos_goal=(h - 1, w - 1),
        pos_agent=np.ones((h, w)),  # распределение стартов по всей карте
        see_obstacle=True,
    )

    # ===================== ОБУЧЕНИЕ DRQN НА 50 СРЕДАХ =====================

    # Векторизированная среда на базе SyncVectorEnv — 50 копий GridWorldEnv
    train_env = GymVectorGridWorldEnv(
        n_envs=50,
        make_env_kwargs=common_env_kwargs,
        base_seed=111,       # базовый seed для карт пола; у каждой копии будет свой (111 + i)
        render_mode=None,
    )

    # Отдельная векторная среда на 5 копий — для оценки/теста
    eval_env = GymVectorGridWorldEnv(
        n_envs=5,
        make_env_kwargs=common_env_kwargs,
        base_seed=111,       # другой базовый сид, чтобы карты могли отличаться от train
        render_mode=None,
    )

    # Гиперпараметры DRQN (можно подправить под задачу)
    agent_kwargs = dict(
        lr=1e-3,
        gamma=0.99,
        seq_len=20,
        burn_in=5,
        batch_size=32,
        buffer_size=500,
        min_episodes=20,     # сколько эпизодов (50-средовых) нужно набрать прежде чем учиться
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=18_000,
        sync_rate=1_000,
        hidden_dim=128,
        avg_window=100,
        optimizer="rmsprop",
        rmsprop_alpha=0.95,
        rmsprop_eps=0.01,
        adadelta_rho=0.9,
        weight_decay=0.0,
        max_grad_norm=10.0,
    )

    # Запуск обучения
    train_drqn_gridworld(
        env=train_env,              # обучение на 50 средах
        max_steps=18_000,           # шаги Lightning-тренера (итерации training_step)
        project="gridworld_drqn",
        run_name="drqn_syncvector_10x10_50envs",
        gif_path="drqn_10x10_syncvector_5envs.gif",  # GIF будет снят на eval_env
        eval_env=eval_env,          # оценка/тест на 5 средах
        **agent_kwargs,
    )
