from __future__ import annotations

import argparse
from typing import Any, Optional, Tuple

import os

import gymnasium as gym
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
import wandb

from src.rl_grid_world.envs.grid_world import GridWorldEnv
from src.rl_grid_world.envs.mnist_grid_world import GridWorldMnistEnv
from src.rl_grid_world.envs.gym_grid_world import GymVectorGridWorldEnv
from src.rl_grid_world.envs.vector_grid_world import VectorGridWorldEnv

from src.rl_grid_world.agents.a2c_agent import A2CAgent, A2CConfig
from src.rl_grid_world.agents.drqn_agent import DRQNLightning
from src.rl_grid_world.utils.generate_wall import generate_walls
from src.rl_grid_world.utils.save_gif import save_episode_gif

def train_a2c_gridworld(
    env: gym.Env,
    config: Optional[A2CConfig] = None,
    project: str = "gridworld_a2c",
    run_name: Optional[str] = None,
    gif_path: Optional[str] = None,
    eval_env: Optional[gym.Env] = None,
    use_wandb: bool = True,
) -> None:
    """
    Обучение A2C-агента.
    """
    if config is None:
        config = A2CConfig()

    run = None
    if use_wandb:
        if wandb.run is not None:
            wandb.finish()

        run = wandb.init(
            project=project,
            name=run_name,
            config={
                "algo": "A2C_LSTM_Batched",
                "gamma": config.gamma,
                "lr": config.lr,
                "entropy_coef": config.entropy_coef,
                "value_coef": config.value_coef,
                "max_grad_norm": config.max_grad_norm,
                "num_episodes": config.num_episodes,
                "max_steps_per_episode": config.max_steps_per_episode,
                "patience": config.patience,
                "min_episodes_before_early_stop": config.min_episodes_before_early_stop,
            },
            reinit=True,
        )

    agent = A2CAgent(env, config, use_wandb=use_wandb)

    agent.train()

    if eval_env is None:
        eval_env = env

    if gif_path is None:
        os.makedirs("gifs", exist_ok=True)
        base_name = (
            run_name or
            (run.name if (use_wandb and run is not None) else "a2c_run")
        )
        gif_path_actual = os.path.join("gifs", f"{base_name}.gif")
    else:
        gif_dir = os.path.dirname(gif_path)
        if gif_dir:
            os.makedirs(gif_dir, exist_ok=True)
        gif_path_actual = gif_path

    fps = 2
    success = False

    print(f"[A2C] Пытаемся записать успешный GIF в {gif_path_actual} ...")
    for attempt in range(10):
        try:
            success = save_episode_gif(
                eval_env,
                agent,
                filename=gif_path_actual,
                fps=fps,
                max_frames=config.max_steps_per_episode,
            )
            print(f"[A2C] Попытка {attempt + 1}/10, success={success}")
            if success:
                break
        except Exception as e:
            print(f"[A2C] Ошибка при записи GIF на попытке {attempt + 1}: {e}")
            break

    if success:
        print(f"[A2C] Успешный GIF сохранён: {gif_path_actual}")
    else:
        print(
            f"[A2C] Не удалось получить успешный эпизод за 10 попыток, "
            f"сохранён последний эпизод: {gif_path_actual}"
        )

    if use_wandb and wandb.run is not None:
        wandb.log({"eval/gif": wandb.Video(gif_path_actual, format="gif")})
        print("[A2C] GIF залогирован в wandb (eval/gif).")

    try:
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)

        base_name = run_name or (run.name if (use_wandb and run is not None) else "a2c")
        ckpt_path = os.path.join(ckpt_dir, f"{base_name}_a2c.pt")

        torch.save(agent.net.state_dict(), ckpt_path)
        print(f"A2C: веса сохранены в {ckpt_path}")

        if use_wandb and wandb.run is not None:
            artifact_name = f"{base_name}_a2c_model"
            artifact = wandb.Artifact(name=artifact_name, type="model")
            artifact.add_file(ckpt_path)
            wandb.run.log_artifact(artifact)
            print(f"A2C: веса залогированы в wandb как artifact '{artifact_name}'")

    except Exception as e:
        print(f"[A2C] Не удалось сохранить/залогировать веса: {e}")

    if use_wandb:
        wandb.finish()


def train_drqn_gridworld(
    env: gym.Env,
    max_steps: int = 50_000,
    project: str = "gridworld_drqn",
    run_name: Optional[str] = None,
    gif_path: Optional[str] = "drqn_eval_episode.gif",
    eval_env: Optional[gym.Env] = None,
    use_wandb: bool = True,
    **agent_kwargs: Any,
) -> None:
    """
    Обучение DRQN‑агента
    """

    agent_kwargs.setdefault("early_stop_min_episodes", 400)
    agent_kwargs.setdefault("early_stop_patience", 250)

    model = DRQNLightning(env=env, **agent_kwargs)

    wandb_logger = None
    if use_wandb:
        if wandb.run is not None:
            wandb.finish()

        wandb_logger = WandbLogger(
            project=project,
            name=run_name,
            log_model=False,
        )

    trainer = pl.Trainer(
        max_steps=max_steps,
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model)

    if model.best_state_dict is not None:
        model.q_net.load_state_dict(model.best_state_dict)
        print(f"[DRQN] Загружены лучшие веса по avg_episode_return={model.best_avg_return:.3f}")
    else:
        print("[DRQN] ВНИМАНИЕ: best_state_dict пуст, используем последние веса.")

    if eval_env is None:
        eval_env = env

    if gif_path is None:
        os.makedirs("gifs", exist_ok=True)
        base_name = run_name or "drqn_run"
        gif_path_actual = os.path.join("gifs", f"{base_name}.gif")
    else:
        gif_dir = os.path.dirname(gif_path)
        if gif_dir:
            os.makedirs(gif_dir, exist_ok=True)
        gif_path_actual = gif_path

    try:
        fps = 2
        print(f"[DRQN] Пытаемся сохранить успешный GIF эпизода в {gif_path_actual} ...")
        success = False
        for attempt in range(10):
            try:
                success = save_episode_gif(
                    eval_env,
                    model,
                    filename=gif_path_actual,
                    fps=fps,
                    max_frames=200,
                )
                print(f"[DRQN] Попытка {attempt + 1}/10, success={success}")
                if success:
                    break
            except Exception as e:
                print(f"[DRQN] Ошибка при записи GIF на попытке {attempt + 1}: {e}")
                break

        if success:
            print(f"[DRQN] Успешный GIF сохранён: {gif_path_actual}")
        else:
            print(
                f"[DRQN] Не удалось получить успешный эпизод за 10 попыток, "
                f"сохранён последний эпизод: {gif_path_actual}"
            )

        if use_wandb and wandb.run is not None:
            wandb.log({"eval/gif": wandb.Video(gif_path_actual, format="gif")})
            print("[DRQN] GIF залогирован в wandb (eval/gif).")

    except Exception as e:
        print(f"[DRQN] Не удалось записать/залогировать GIF: {e}")

    if use_wandb and wandb.run is not None:
        try:
            ckpt_dir = "checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            base_name = run_name or "drqn_run"
            ckpt_path = os.path.join(ckpt_dir, f"{base_name}_drqn.pt")
            torch.save(model.q_net.state_dict(), ckpt_path)
            print(f"[DRQN] Веса q_net сохранены в {ckpt_path}")

            artifact_name = f"{base_name}_drqn_model"
            artifact = wandb.Artifact(name=artifact_name, type="model")
            artifact.add_file(ckpt_path)
            wandb.run.log_artifact(artifact)
            print(f"[DRQN] Веса залогированы в wandb как artifact '{artifact_name}'")
        except Exception as e:
            print(f"[DRQN] Не удалось залогировать веса DRQN: {e}")

    if use_wandb:
        wandb.finish()


def build_envs(
    env_type: str,
    h: int,
    w: int,
    n_colors: int,
    obstacle_ratio: float,
    seed: int,
    image_size: int,
    digits_mode: str,
    n_envs_train: int,
    n_envs_eval: int,
    see_obstacle: bool,
    step_reward: float = 0.0,   # <-- добавлен default
):
    """
      1) "onehot"   : обычная GridWorldEnv с one-hot наблюдениями.
      2) "mnist"    : GridWorldMnistEnv (наблюдения = картинка цифры).
      3) "gym_vec"  : GymVectorGridWorldEnv (SyncVectorEnv базовая векторизация).
      4) "np_vec"   : VectorGridWorldEnv (полностью numpy-векторизованная).

    Параметр step_reward передаётся во все GridWorld‑среды.
    """

    obstacle_mask = generate_walls(
        h=h,
        w=w,
        obstacle_ratio=obstacle_ratio,
        start=(0, 0),
        goal=(h - 1, w - 1),
    )

    common_env_kwargs = dict(
        h=h,
        w=w,
        n_colors=n_colors,
        obstacle_mask=obstacle_mask,
        pos_goal=(h - 1, w - 1),
        pos_agent=np.ones((h, w)),
        see_obstacle=see_obstacle,
        step_reward=step_reward,
    )

    if env_type == "onehot":
        train_env = GridWorldEnv(
            **common_env_kwargs,
            seed=seed,
            render_mode=None,
        )
        eval_env = GridWorldEnv(
            **common_env_kwargs,
            seed=seed,
            render_mode=None,
        )

    elif env_type == "mnist":
        train_env = GridWorldMnistEnv(
            **common_env_kwargs,
            seed=seed,
            mnist_root="./mnist_data",
            image_size=image_size,
            flatten=True,
            render_mode=None,
            digit_render_mode=digits_mode,
        )
        eval_env = GridWorldMnistEnv(
            **common_env_kwargs,
            seed=seed,
            mnist_root="./mnist_data",
            image_size=image_size,
            flatten=True,
            render_mode=None,
            digit_render_mode=digits_mode,
        )

    elif env_type == "gym_vec":
        train_env = GymVectorGridWorldEnv(
            n_envs=n_envs_train,
            make_env_kwargs=common_env_kwargs, 
            base_seed=seed,
            render_mode=None,
        )
        eval_env = GymVectorGridWorldEnv(
            n_envs=n_envs_eval,
            make_env_kwargs=common_env_kwargs,
            base_seed=seed + 1000,
            render_mode=None,
        )

    elif env_type == "np_vec":
        train_env = VectorGridWorldEnv(
            **common_env_kwargs,
            render_mode=None,
            n_envs=n_envs_train,
            seed=seed,
        )
        eval_env = VectorGridWorldEnv(
            **common_env_kwargs,
            render_mode=None,
            n_envs=n_envs_eval,
            seed=seed + 1000,
        )

    else:
        raise ValueError(
            f"Неизвестный тип среды env_type={env_type}, "
            f"ожидалось onehot|mnist|gym_vec|np_vec"
        )

    return train_env, eval_env

def main():
    parser = argparse.ArgumentParser(description="Train A2C / DRQN on GridWorld family")

    parser.add_argument("--name", type=str, default="a2c_run", help="Run name (wandb и имя GIF)")
    parser.add_argument("--algo", type=str, default="a2c", choices=["a2c", "drqn"], help="Алгоритм обучения")
    parser.add_argument(
        "--env_type",
        type=str,
        default="onehot",
        choices=["onehot", "mnist", "gym_vec", "np_vec"],
        help="Тип среды: onehot | mnist | gym_vec | np_vec",
    )

    parser.add_argument("--h", type=int, default=10, help="Высота поля")
    parser.add_argument("--w", type=int, default=10, help="Ширина поля")
    parser.add_argument("--episodes", type=int, default=2000, help="Число эпизодов (для A2C)")
    parser.add_argument("--obstacle_ratio", type=float, default=0.1, help="Доля препятствий")
    parser.add_argument("--n_colors", type=int, default=4, help="Количество цветов пола")
    parser.add_argument("--seed", type=int, default=111, help="Seed для карты пола/старта")
    parser.add_argument("--see_obstacle",  action="store_true", help="Видит ли препятствия")

    parser.add_argument(
        "--step_reward",
        type=float,
        default=0.0,
        help="Награда/штраф за шаг (например, -0.01, чтобы ускорить достижение цели)",
    )

    parser.add_argument("--no-wandb", action="store_true", help="Отключить логирование в wandb")

    parser.add_argument("--gif", type=str, default=None, help="Путь к GIF (если None, сформировать автоматически)")

    parser.add_argument("--image_size", type=int, default=14, help="Размер MNIST-картинки (12..28)")
    parser.add_argument(
        "--digits",
        type=str,
        default="text",
        choices=["text", "mnist"],
        help="Режим визуализации цифр в MNIST-среде: text или mnist",
    )

    parser.add_argument("--n_envs_train", type=int, default=16, help="Число под-сред при обучении (vector env)")
    parser.add_argument("--n_envs_eval", type=int, default=2, help="Число под-сред при оценке (vector env)")

    parser.add_argument("--max_steps", type=int, default=50_000, help="max_steps для DRQN (Trainer)")

    args = parser.parse_args()

    use_wandb = not args.no_wandb

    train_env, eval_env = build_envs(
        env_type=args.env_type,
        h=args.h,
        w=args.w,
        n_colors=args.n_colors,
        obstacle_ratio=args.obstacle_ratio,
        seed=args.seed,
        image_size=args.image_size,
        digits_mode=args.digits,
        n_envs_train=args.n_envs_train,
        n_envs_eval=args.n_envs_eval,
        see_obstacle=args.see_obstacle,
        step_reward=args.step_reward,
    )

    if args.gif is not None:
        gif_path = args.gif
    else:
        gif_env_tag = args.env_type
        gif_algo_tag = args.algo
        gif_path = f"{args.name}_{gif_algo_tag}_{gif_env_tag}_{args.h}x{args.w}.gif"

    if args.algo == "a2c":
        cfg = A2CConfig(
            gamma=0.99,
            lr=1e-3,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            num_episodes=args.episodes,
            max_steps_per_episode=args.h * args.w,
            print_every=100,
            avg_window=100,
            patience=20,
            min_episodes_before_early_stop=50,
        )

        train_a2c_gridworld(
            env=train_env,
            config=cfg,
            project="gridworld_a2c",
            run_name=args.name,
            gif_path=gif_path,
            eval_env=eval_env,
            use_wandb=use_wandb,
        )

    elif args.algo == "drqn":
        drqn_kwargs = dict(
            lr=1e-3,
            gamma=0.99,
            seq_len=20,
            burn_in=5,
            batch_size=32,
            buffer_size=500,
            min_episodes=20,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=20_000,
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

        train_drqn_gridworld(
            env=train_env,
            max_steps=args.max_steps,
            project="gridworld_drqn",
            run_name=args.name,
            gif_path=gif_path,
            eval_env=eval_env,
            use_wandb=use_wandb,
            **drqn_kwargs,
        )

    else:
        raise ValueError(f"Неизвестный алгоритм {args.algo}")


if __name__ == "__main__":
    main()