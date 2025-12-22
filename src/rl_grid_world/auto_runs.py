from __future__ import annotations

from typing import Dict, List, Optional

from src.rl_grid_world.train import (
    build_envs,
    train_a2c_gridworld,
    train_drqn_gridworld,
)
from src.rl_grid_world.agents.a2c_agent import A2CConfig


def make_cli_command(
    sid: str,
    algo: str,
    env_type: str,
    h: int,
    w: int,
    obstacle_ratio: float,
    n_colors: int,
    see_obstacle: bool,
    digits_mode: Optional[str],
    n_envs_train: int,
    n_envs_eval: int,
    episodes: int,
    max_steps_drqn: int,
    name: str,
) -> str:
    """
    Строит "виртуальную" CLI-команду, соответствующую данному запуску.
    """
    base = [
        "python -m src.rl_grid_world.train",
        f'--name "{name}"',
        f"--algo {algo}",
        f"--env_type {env_type}",
        f"--h {h}",
        f"--w {w}",
        f"--obstacle_ratio {obstacle_ratio}",
        f"--n_colors {n_colors}",
    ]

    if see_obstacle:
        base.append("--see_obstacle")
    else:
        base.append("--no-see_obstacle")

    if env_type == "mnist" and digits_mode is not None:
        base.append(f"--digits {digits_mode}")
        base.append(f"--image_size 14")

    if env_type in ("gym_vec", "np_vec"):
        base.append(f"--n_envs_train {n_envs_train}")
        base.append(f"--n_envs_eval {n_envs_eval}")

    if algo == "a2c":
        base.append(f"--episodes {episodes}")
    else:  # drqn
        base.append(f"--max_steps {max_steps_drqn}")

    return " ".join(base)


def run_suite(
    project: str = "gridworld_suite",
    n_envs_train: int = 50,
    n_envs_eval: int = 5,
):
    """
    Запускает ВСЕ комбинации сценариев / типов сред / алгоритмов / see_obstacle / digits_mode.
    """

    SCENARIOS: List[Dict] = [
        dict(
            id="s1",
            h=5,
            w=5,
            obstacle_ratio=0.0,
            n_colors=25,
            env_types=["onehot", "gym_vec", "np_vec"], 
            episodes=500,
            max_steps_drqn=20_000,
        ),
        dict(
            id="s2",
            h=5,
            w=5,
            obstacle_ratio=0.0,
            n_colors=5,
            env_types=["onehot", "mnist", "gym_vec", "np_vec"],
            episodes=2000,
            max_steps_drqn=50_000,
        ),
        dict(
            id="s3",
            h=10,
            w=10,
            obstacle_ratio=0.1,
            n_colors=7,
            env_types=["onehot", "mnist", "gym_vec", "np_vec"],
            episodes=3000,
            max_steps_drqn=80_000,
        ),
        dict(
            id="s4",
            h=10,
            w=10,
            obstacle_ratio=0.1,
            n_colors=4,
            env_types=["onehot", "mnist", "gym_vec", "np_vec"],
            episodes=3000,
            max_steps_drqn=80_000,
        ),
    ]

    ALGOS = ["a2c", "drqn"]
    SEE_OPTIONS = [True, False]

    MNIST_DIGITS_MODES = ["text", "mnist"]  

    DRQN_COMMON_KWARGS = dict(
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

    results = []  # для чек-листа

    for scenario in SCENARIOS:
        sid = scenario["id"]
        h = scenario["h"]
        w = scenario["w"]
        obstacle_ratio = scenario["obstacle_ratio"]
        n_colors = scenario["n_colors"]
        env_types = scenario["env_types"]
        episodes = scenario["episodes"]
        max_steps_drqn = scenario["max_steps_drqn"]

        for env_type in env_types:
            if env_type == "mnist":
                digits_modes = MNIST_DIGITS_MODES
            else:
                digits_modes = [None]

            for digits_mode in digits_modes:
                for see_obstacle in SEE_OPTIONS:
                    for algo in ALGOS:
                        see_str = "see" if see_obstacle else "nosee"
                        digits_str = f"_{digits_mode}" if digits_mode is not None else ""
                        run_name = f"{sid}_{algo}_{env_type}{digits_str}_{see_str}"

                        cli_cmd = make_cli_command(
                            sid=sid,
                            algo=algo,
                            env_type=env_type,
                            h=h,
                            w=w,
                            obstacle_ratio=obstacle_ratio,
                            n_colors=n_colors,
                            see_obstacle=see_obstacle,
                            digits_mode=digits_mode,
                            n_envs_train=n_envs_train,
                            n_envs_eval=n_envs_eval,
                            episodes=episodes,
                            max_steps_drqn=max_steps_drqn,
                            name=run_name,
                        )

                        print(f"\n=== Запуск {run_name} ===")
                        print(f"CLI: {cli_cmd}")

                        status = "success"
                        error_msg = ""

                        try:
                            # создаём среды
                            train_env, eval_env = build_envs(
                                env_type=env_type,
                                h=h,
                                w=w,
                                n_colors=n_colors,
                                obstacle_ratio=obstacle_ratio,
                                seed=111,
                                image_size=14,
                                digits_mode=digits_mode if digits_mode is not None else "text",
                                n_envs_train=n_envs_train,
                                n_envs_eval=n_envs_eval,
                                see_obstacle=see_obstacle,
                            )

                            if algo == "a2c":
                                project_name = f"{project}_a2c"
                                cfg = A2CConfig(
                                    gamma=0.99,
                                    lr=1e-3,
                                    entropy_coef=0.01,
                                    value_coef=0.5,
                                    max_grad_norm=0.5,
                                    num_episodes=episodes,
                                    max_steps_per_episode=h * w,
                                    print_every=100,
                                    avg_window=100,
                                )

                                train_a2c_gridworld(
                                    env=train_env,
                                    config=cfg,
                                    project=project,
                                    run_name=run_name,
                                    gif_path=None,
                                    eval_env=eval_env,
                                    use_wandb=True,
                                )

                            else:  # DRQN
                                project_name = f"{project}_drqn"
                                train_drqn_gridworld(
                                    env=train_env,
                                    max_steps=max_steps_drqn,
                                    project=project,
                                    run_name=run_name,
                                    gif_path=None,
                                    eval_env=eval_env,
                                    use_wandb=True,
                                    **DRQN_COMMON_KWARGS,
                                )

                        except Exception as e:
                            status = "failed"
                            error_msg = str(e)
                            print(f"[ОШИБКА] Запуск {run_name} завершился с ошибкой:")
                            print(f"         {e}")
                            print(f"         Команда: {cli_cmd}")

                        results.append(
                            dict(
                                run_name=run_name,
                                scenario=sid,
                                algo=algo,
                                env_type=env_type,
                                digits_mode=digits_mode,
                                see_obstacle=see_obstacle,
                                status=status,
                                error=error_msg,
                                cli=cli_cmd,
                            )
                        )


    print("\nИТОГОВЫЙ ЧЕК-ЛИСТ")

    total = len(results)
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = total - succeeded

    print(f"Всего запусков:     {total}")
    print(f"Успешно завершено:  {succeeded}")
    print(f"С ошибками:         {failed}\n")

    print("Подробности:")
    for r in results:
        mark = "[OK]" if r["status"] == "success" else "[FAIL]"
        desc = f"{r['run_name']} | scen={r['scenario']} | algo={r['algo']} | env={r['env_type']}"
        if r["env_type"] == "mnist":
            desc += f" | digits={r['digits_mode']}"
        desc += f" | see_obstacle={r['see_obstacle']}"
        print(f"{mark} {desc}")
        if r["status"] == "failed":
            print(f"     error: {r['error']}")
            print(f"     cmd:   {r['cli']}")

if __name__ == "__main__":
    run_suite(project="gridworld_experiment2", n_envs_train=20, n_envs_eval=1)