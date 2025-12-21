from __future__ import annotations

import numpy as np

from src.rl_grid_world.envs.mnist_grid_world import GridWorldMnistEnv
from src.rl_grid_world.agents.a2c_agent import A2CAgent, A2CConfig
from src.rl_grid_world.utils.generate_wall import generate_walls
from src.rl_grid_world.utils.save_gif import save_episode_gif  # твой общий save_episode_gif


if __name__ == "__main__":
    h, w = 10, 10

    obstacle_mask = generate_walls(
        h=h,
        w=w,
        obstacle_ratio=0.1,
        start=(0, 0),
        goal=(h - 1, w - 1),
    )

    # Общие параметры для MNIST-среды
    common_env_kwargs = dict(
        h=h,
        w=w,
        n_colors=4,                 # максимум 7 для MNIST-варианта
        obstacle_mask=obstacle_mask,
        pos_goal=(h - 1, w - 1),
        pos_agent=np.ones((h, w)),  # равномерное распределение стартов
        see_obstacle=True,
        seed=111,                   # фиксируем раскладку пола
        mnist_root="./mnist_data",
        image_size=14,              # 14x14 картинка из MNIST (центр-кроп)
        flatten=True,               # чтобы A2C видел 1D-вектор
        render_mode=None,
    )

    # Среда для обучения
    train_env = GridWorldMnistEnv(**common_env_kwargs)

    # Отдельная среда для визуализации / GIF (тот же seed)
    eval_env = GridWorldMnistEnv(**common_env_kwargs)

    # Конфиг A2C
    cfg = A2CConfig(
        gamma=0.99,
        lr=1e-3,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        num_episodes=2000,
        max_steps_per_episode=h * w,
        print_every=100,
        avg_window=100,
    )

    # Создаём и обучаем A2C-агента
    agent = A2CAgent(
        env=train_env,
        config=cfg,
        use_wandb=True,    # можешь выключить, если не нужен wandb
    )
    agent.train()

    # Сохраняем GIF с ПОЛЕМ (цветная карта + цифры в клетках) из eval_env
    save_episode_gif(
        env=eval_env,
        model=agent,
        filename="a2c_10x10_gridworld_mnist_field.gif",
        fps=2,
        max_frames=cfg.max_steps_per_episode,
        hold_last=10,
    )