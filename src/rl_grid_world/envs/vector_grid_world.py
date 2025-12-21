from typing import Optional, Union, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches


class VectorGridWorldEnv(gym.Env):
    r"""
    Векторизованная GridWorld‑среда.

    Внутри хранится 'n_envs' независимых копий одиночной среды.
    На каждый шаг подаётся действие для КАЖДОЙ копии (actions[i]),
    а на выходе возвращаются массивы по всем под‑средам.

    Интерфейс шага:

        obs, reward, terminated, truncated, info = env.step(actions)

    где:

        actions: np.ndarray shape (n_envs,) или скаляр int (бродкастится);

        obs:        np.ndarray shape (n_envs, single_obs_dim),
        reward:     np.ndarray shape (n_envs,), значения step_reward или goal_reward;
        terminated: np.ndarray shape (n_envs,), True если env достигла цели;
        truncated:  np.ndarray shape (n_envs,), True если превышен лимит шагов;
        info: dict с полями:
            - "reward_per_env":      reward;
            - "terminated_per_env":  terminated;
            - "truncated_per_env":   truncated.

    Каждый эпизод под‑среды:
    - начинается в reset();
    - заканчивается, когда агент дошёл до цели или превысил лимит шагов.

    Векторная логика (движение, коллизии, награды) реализована целиком через numpy.

    Параметры награды:
      - step_reward: награда/штраф за обычный шаг (по умолчанию 0.0);
      - goal_reward: награда за достижение цели (по умолчанию 1.0).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        h: int = 10,
        w: int = 10,
        obstacle_mask: Optional[np.ndarray] = None,
        n_colors: int = 1,
        pos_goal: Union[Tuple[int, int], np.ndarray] = (0, 0),
        pos_agent: Union[Tuple[int, int], np.ndarray] = (1, 1),
        see_obstacle: bool = True,
        render_mode: Optional[str] = None,
        n_envs: int = 2,
        seed: Optional[int] = None,
        step_reward: float = 0.0,
        goal_reward: float = 1.0,
    ):
        super().__init__()

        if n_envs <= 0:
            raise ValueError("n_envs must be >= 1")
        self.n_envs = int(n_envs)

        self.h = int(h)
        self.w = int(w)
        self.n_colors = int(n_colors)
        self.pos_goal = np.array(pos_goal, dtype=int)
        self.start_pos = pos_agent
        self.max_steps = self.h * self.w
        self.see_obstacle = bool(see_obstacle)
        self.render_mode = render_mode

        self.seed = seed

        # shaping‑параметры награды
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)

        if np.any(self.pos_goal < 0) or np.any(self.pos_goal >= np.array([self.h, self.w])):
            raise ValueError(f"pos_goal {self.pos_goal} вне границ сетки {self.h}x{self.w}")

        if obstacle_mask is None:
            self.obstacle_mask = np.zeros((self.h, self.w), dtype=bool)
        else:
            mask = np.array(obstacle_mask, dtype=bool)
            if mask.shape == (self.w, self.h) and self.w != self.h:
                mask = mask.T
            if mask.shape != (self.h, self.w):
                raise ValueError(
                    f"obstacle_mask имеет форму {mask.shape}, а должна быть ({self.h}, {self.w})"
                )
            self.obstacle_mask = mask

        self.feature_wall = self.n_colors
        self.feature_obstacle = self.n_colors + 1
        self.feature_goal = self.n_colors + 2
        self.single_obs_dim = self.feature_goal + 1

        self.action_space = spaces.Discrete(4)

        # Для простоты оставляем observation_space как (n_envs, single_obs_dim),
        # хотя формально это батч из n_envs наблюдений по single_obs_dim каждый.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_envs, self.single_obs_dim),
            dtype=np.float32,
        )

        # Сетка: для каждого env своя карта (h+2, w+2)
        self.grid = np.zeros((self.n_envs, self.h + 2, self.w + 2), dtype=int)

        # Позиции агента и "видимая" позиция
        self.agent_pos = np.zeros((self.n_envs, 2), dtype=int)
        self.see_pos = np.zeros((self.n_envs, 2), dtype=int)

        # Счётчик шагов и done‑флаг для каждой под‑среды
        self.steps = np.zeros(self.n_envs, dtype=int)
        self.done = np.zeros(self.n_envs, dtype=bool)

        # Векторы движения по действиям (0..3)
        self._move_vectors = np.array(
            [
                [-1, 0],   # вверх
                [0, 1],    # вправо
                [1, 0],    # вниз
                [0, -1],   # влево
            ],
            dtype=int,
        )

    def _init_grid(self, seed: Optional[int]) -> None:
        """
        Инициализация всех сеток:
        - стены по периметру,
        - СЛУЧАЙНЫЕ типы пола внутри (одна карта), далее переносим на все envs,
        - препятствия по obstacle_mask,
        - цель в pos_goal для каждой из n_envs.
        """
        grid_seed = self.seed if self.seed is not None else seed
        rng = np.random.default_rng(grid_seed)

        self.grid.fill(self.feature_wall)

        random_floors_single = rng.integers(
            low=0,
            high=self.n_colors,
            size=(self.h, self.w),
            dtype=int,
        )

        inner_single = np.where(
            self.obstacle_mask,
            self.feature_obstacle,
            random_floors_single,
        )  # (h, w)

        inner_area = np.broadcast_to(inner_single, (self.n_envs, self.h, self.w))

        self.grid[:, 1:-1, 1:-1] = inner_area

        goal_r, goal_c = (self.pos_goal + 1).astype(int)
        env_idx = np.arange(self.n_envs, dtype=int)
        self.grid[env_idx, goal_r, goal_c] = self.feature_goal

    def _sample_start_positions(self):
        """
        Выбор стартовой позиции для каждого env (агента).
        self.start_pos:
          - если вектор длины 2: фиксированные координаты (h,w) для всех env;
          - если матрица (h, w): распределение вероятностей по клеткам.
        """
        pos_config = np.array(self.start_pos)

        if pos_config.ndim == 1:
            base = pos_config.astype(int) + 1
            self.agent_pos[:] = base[None, :]
        else:
            probs = np.array(pos_config, dtype=np.float32)

            if probs.shape == (self.w, self.h) and self.w != self.h:
                probs = probs.T

            if probs.shape != (self.h, self.w):
                raise ValueError(
                    f"start_pos как распределение имеет форму {probs.shape}, "
                    f"а должна быть ({self.h}, {self.w})"
                )

            probs = probs.copy()
            probs[self.obstacle_mask] = 0.0
            probs[tuple(self.pos_goal)] = 0.0

            total = probs.sum()
            if total > 0:
                probs /= total
                flat_probs = probs.ravel()

                flat_idx = self.np_random.choice(
                    flat_probs.size,
                    size=self.n_envs,
                    p=flat_probs,
                )
                rr, cc = np.unravel_index(flat_idx, probs.shape)  # (h, w)

                self.agent_pos[:, 0] = rr + 1
                self.agent_pos[:, 1] = cc + 1
            else:
                self.agent_pos[:] = np.array([[1, 1]], dtype=int)

        self.see_pos[:] = self.agent_pos

    def _get_obs(self) -> np.ndarray:
        """
        Возвращает наблюдение shape (n_envs, single_obs_dim):
        - для каждой под‑среды берём значение клетки в self.grid[env, see_pos[env]],
        - кодируем one‑hot.
        """
        env_idx = np.arange(self.n_envs, dtype=int)
        rs = self.see_pos[:, 0]
        cs = self.see_pos[:, 1]

        cell_vals = self.grid[env_idx, rs, cs]  # (n_envs,)
        obs = np.zeros((self.n_envs, self.single_obs_dim), dtype=np.float32)
        obs[env_idx, cell_vals] = 1.0
        return obs

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Сбрасывает ВСЕ n_envs копий среды.

        Возвращает:
          obs:  np.ndarray shape (n_envs, single_obs_dim)
          info: dict (пока пустой).
        """
        super().reset(seed=seed)

        self._init_grid(seed)
        self.steps[:] = 0
        self.done[:] = False

        self._sample_start_positions()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, actions):
        """
        Выполняет один векторизованный шаг.

        actions:
          - int: одно и то же действие для всех envs;
          - или np.ndarray shape (n_envs,) со своими действиями.

        Возвращает:
          obs:        (n_envs, single_obs_dim)
          reward:     (n_envs,) float32
          terminated: (n_envs,) bool
          truncated:  (n_envs,) bool
          info: dict с теми же векторами.
        """

        actions_arr = np.asarray(actions, dtype=int)
        if actions_arr.ndim == 0:
            actions_arr = np.full(self.n_envs, int(actions_arr), dtype=int)
        elif actions_arr.shape != (self.n_envs,):
            raise ValueError(
                f"actions должен быть скаляром или shape ({self.n_envs},), "
                f"получено {actions_arr.shape}"
            )

        if np.any((actions_arr < 0) | (actions_arr >= 4)):
            raise ValueError("Каждое действие должно быть в диапазоне 0..3")

        active = np.logical_not(self.done)
        env_idx = np.arange(self.n_envs, dtype=int)

        moves = self._move_vectors[actions_arr]  # (n_envs, 2)

        target_pos = self.agent_pos + moves  # (n_envs, 2)
        tr = target_pos[:, 0]
        tc = target_pos[:, 1]

        cell_vals = self.grid[env_idx, tr, tc]  # (n_envs,)

        is_wall = cell_vals == self.feature_wall
        is_obstacle = cell_vals == self.feature_obstacle
        is_goal = cell_vals == self.feature_goal
        is_blocked = np.logical_or(is_wall, is_obstacle)

        can_move = np.logical_and(np.logical_not(is_blocked), active)
        blocked_active = np.logical_and(is_blocked, active)

        new_agent_pos = self.agent_pos.copy()
        new_see_pos = self.see_pos.copy()

        new_agent_pos[can_move] = target_pos[can_move]
        new_see_pos[can_move] = target_pos[can_move]

        if self.see_obstacle:
            new_see_pos[blocked_active] = target_pos[blocked_active]
        else:
            new_see_pos[blocked_active] = self.agent_pos[blocked_active]

        # увеличиваем шаги только для активных под‑сред
        self.steps[active] += 1

        reached_goal = np.logical_and(is_goal, active)
        terminated_this_step = reached_goal.copy()

        truncated_this_step = np.logical_and(self.steps >= self.max_steps, active)

        # Shaping‑награда:
        #   - step_reward за каждый шаг для живых env;
        #   - goal_reward при достижении цели на этом шаге;
        #   - env, которые уже done, не получают ничего.
        reward_per_env = np.zeros(self.n_envs, dtype=np.float32)
        reward_per_env[active] = self.step_reward
        reward_per_env[reached_goal] = self.goal_reward

        self.done = np.logical_or(
            self.done,
            np.logical_or(terminated_this_step, truncated_this_step),
        )

        self.agent_pos = new_agent_pos
        self.see_pos = new_see_pos

        obs = self._get_obs()

        info = {
            "reward_per_env": reward_per_env.copy(),
            "terminated_per_env": terminated_this_step.copy(),
            "truncated_per_env": truncated_this_step.copy(),
        }

        return obs, reward_per_env, terminated_this_step, truncated_this_step, info

    def render(self):
        r"""
        (render_mode='human'),
        (render_mode='rgb_array').
        """
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            self._render_map(return_rgb_array=False)
        elif self.render_mode == "rgb_array":
            return self._render_map(return_rgb_array=True)
        else:
            raise ValueError(f"Неизвестный render_mode: {self.render_mode}")

    def _build_colormap(self):
        """
        Строит colormap для визуализации сетки.
        """

        base_cmap = plt.get_cmap("Set3")

        palette_colors = [base_cmap(i % base_cmap.N) for i in range(self.n_colors)]
        palette_colors.append("black")      # стена
        palette_colors.append("firebrick")  # препятствие
        palette_colors.append("green")      # цель

        my_cmap = colors.ListedColormap(palette_colors)
        bounds = np.arange(-0.5, self.feature_goal + 1.5, 1)
        norm = colors.BoundaryNorm(bounds, my_cmap.N)

        patches = [mpatches.Patch(color="gold", label="Agent")]
        for i in range(self.n_colors):
            patches.append(mpatches.Patch(color=palette_colors[i], label=f"Floor {i}"))
        patches.append(mpatches.Patch(color="black", label="Wall"))
        patches.append(mpatches.Patch(color="firebrick", label="Obstacle"))
        patches.append(mpatches.Patch(color="green", label="Goal"))

        return my_cmap, norm, patches

    def _render_map(self, return_rgb_array: bool = False):
        """
        Отрисовывает все n_envs копий среды в виде сетки сабплотов.

        Если return_rgb_array=True, возвращает np.ndarray (H, W, 3).
        """

        my_cmap, norm, legend_patches = self._build_colormap()

        n = self.n_envs
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
        )
        axes = np.atleast_1d(axes).ravel()

        for i in range(n):
            ax = axes[i]
            grid_i = self.grid[i]

            ax.imshow(grid_i, cmap=my_cmap, norm=norm, origin="upper")

            ay, ax_coord = self.agent_pos[i]
            ax.scatter(
                ax_coord,
                ay,
                c="gold",
                s=200,
                marker="*",
                edgecolors="black",
                zorder=10,
            )

            ax.set_xticks(np.arange(-0.5, self.w + 2, 1))
            ax.set_yticks(np.arange(-0.5, self.h + 2, 1))
            ax.grid(which="major", color="gray", linestyle="-", linewidth=1)
            ax.tick_params(
                axis="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            ax.set_title(f"Env {i}")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
        )
        fig.suptitle(f"GridWorld {self.h}x{self.w}, n_envs={self.n_envs}")
        fig.tight_layout()

        if return_rgb_array:
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf, dtype=np.uint8)[..., :3]
            plt.close(fig)
            return img
        else:
            plt.show()