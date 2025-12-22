"""
Модуль реализует среду GridWorld для задач обучения с подкреплением.

Среда представляет собой прямоугольное поле с клетками разных типов: стены, препятствия,
цель и пол с разными цветами. Агент может перемещаться по полю и получать наблюдения
о текущей клетке. Цель агента — достичь клетки с целью.

Пример использования:

    >>> env = GridWorldEnv(h=5, w=5, n_colors=3, render_mode='human')
    >>> obs, _ = env.reset(seed=42)
    >>> print(obs)  # One-hot вектор наблюдения
    [0. 0. 0. 1. 0. 0. 0.]
    >>> action = 2  # Движение вниз
    >>> obs, reward, terminated, truncated, info = env.step(action)
    >>> env.render()
    >>> if terminated:
    ...     print(f'Цель достигнута! Награда: {reward}')

Note:

    - Среда использует одномерные one-hot векторы для наблюдений.
    - Координаты в сетке начинаются с (0, 0) в верхнем левом углу.
    - Использование `see_obstacle=True` позволяет получать информацию о препятствиях.
"""

from typing import Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    r"""Среда-сетка для задач обучения с подкреплением.

    Среда представляет собой двумерную сетку размером h*w, окруженную стенами.
    Внутри сетки могут располагаться препятствия и одна клетка-цель.
    Пол может иметь разные цвета (типы), задаваемые параметром n_colors.

    Attributes:
        h (int): Высота внутренней сетки (без стен).
        w (int): Ширина внутренней сетки (без стен).
        n_colors (int): Количество типов пола (0 до n_colors-1).
        pos_goal (np.ndarray): Позиция цели в координатах внутренней сетки.
        start_pos (Union[Tuple[int, int], np.ndarray]):
            Начальная позиция агента или распределение вероятностей.
        see_obstacle (bool): Определяет, видны ли препятствия при наблюдении.
        render_mode (Optional[str]): Режим отрисовки ('human' или 'rgb_array').

        obstacle_mask (np.ndarray): Булева маска препятствий размером (h, w).
        feature_wall (int): Значение признака для стен в векторе наблюдений.
        feature_obstacle (int): Значение признака для препятствий.
        feature_goal (int): Значение признака для цели.

        grid (np.ndarray): Внутренняя сетка размером (h+2, w+2) с границами-стенами.
        agent_pos (np.ndarray): Текущая позиция агента в координатах сетки со стенами.
        see_pos (np.ndarray): Позиция, которую "видит" агент для наблюдений.

        seed (int): Зерно для генерации среды, для возможности повторно сгенерировать карту.

        step_reward (float): Награда/штраф за обычный шаг (по умолчанию 0.0).
        goal_reward (float): Награда за достижение цели (по умолчанию 1.0).

    Action Space:
        Пространство действий — дискретное с 4 направлениями:

        - 0: вверх     ( [-1,  0] )
        - 1: вправо    ( [ 0,  1] )
        - 2: вниз      ( [ 1,  0] )
        - 3: влево     ( [ 0, -1] )

    Observation Space:
        Пространство наблюдений — вещественный вектор размерности
        (n_colors + 3), закодированный как one-hot вектор,
        где 3 дополнительных признака соответствуют стене, препятствию и цели.

        $$\text{shape} = (\text{n_{colors}} + 3,)$$

    Reward:
        По умолчанию:
          - Награда goal_reward (1.0) при достижении цели (клетки GOAL),
          - Награда step_reward (0.0) в остальных случаях.

        Можно включить shaping, задав отрицательный step_reward
        (например, -0.01, чтобы поощрять быстрый путь).

    Episode Termination:
        Эпизод завершается, когда агент достигает клетки с целью или же когда превышено выделенное
        количество шагов на эпизодов (по умолчанию берется w*h)
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
        seed: Optional[int] = None,
        step_reward: float = -0.01,
        goal_reward: float = 1.0,
    ):
        super().__init__()

        self.h = int(h)
        self.w = int(w)
        self.n_colors = int(n_colors)
        self.pos_goal = np.array(pos_goal, dtype=int)
        self.start_pos = pos_agent
        self.max_steps = self.h * self.w
        self.see_obstacle = bool(see_obstacle)
        self.render_mode = render_mode

        self.seed = seed

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

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.feature_goal + 1,),
            dtype=np.float32,
        )

        self.grid = np.zeros((self.h + 2, self.w + 2), dtype=int)

        self.agent_pos = np.array([0, 0], dtype=int)
        self.see_pos = np.array([0, 0], dtype=int)

        self.current_step = 0

        # === ИСПРАВЛЕНИЕ: сохраняем базовую карту ===
        self._base_grid: Optional[np.ndarray] = None
        self._grid_seed_used: Optional[int] = None

        # Если seed задан при создании, сразу генерируем и сохраняем карту
        if self.seed is not None:
            self._generate_and_save_grid(self.seed)

    def _generate_and_save_grid(self, grid_seed: int) -> None:
        """Генерирует карту один раз и сохраняет её как базовую."""
        rng = np.random.default_rng(grid_seed)

        self.grid.fill(self.feature_wall)

        random_floors = rng.integers(
            low=0,
            high=self.n_colors,
            size=(self.h, self.w),
            dtype=int,
        )

        inner_area = np.where(
            self.obstacle_mask,
            self.feature_obstacle,
            random_floors,
        )

        self.grid[1:-1, 1:-1] = inner_area

        goal_r, goal_c = (self.pos_goal + 1).astype(int)
        self.grid[goal_r, goal_c] = self.feature_goal

        # Сохраняем карту
        self._base_grid = self.grid.copy()
        self._grid_seed_used = grid_seed

    def _init_grid(self, seed: Optional[int]) -> None:
        r"""Инициализирует внутреннюю сетку среды.

        Если карта уже была сгенерирована и сохранена, восстанавливает её.
        Иначе генерирует новую карту и сохраняет.
        """
        # Определяем seed для карты
        grid_seed = self.seed if self.seed is not None else seed

        # Если карта уже сохранена и seed совпадает (или self.seed зафиксирован),
        # просто восстанавливаем
        if self._base_grid is not None:
            if self.seed is not None or grid_seed == self._grid_seed_used:
                self.grid[:] = self._base_grid
                return

        # Если seed не задан ни в конструкторе, ни в reset, генерируем случайный
        if grid_seed is None:
            grid_seed = np.random.randint(0, 2**31 - 1)

        # Генерируем и сохраняем карту
        self._generate_and_save_grid(grid_seed)

    def _get_obs(self) -> np.ndarray:
        r"""Возвращает наблюдение агента в виде one-hot вектора."""
        obs_vec = np.zeros((self.feature_goal + 1,), dtype=np.float32)
        val = self.grid[tuple(self.see_pos)]
        obs_vec[val] = 1.0
        return obs_vec

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Перезапуск среды.

        seed:
          - передаётся в super().reset(seed=seed) → инициализирует self.np_random;
          - используется в _init_grid ТОЛЬКО если self.seed is None и карта ещё не сохранена.

        Возвращает:
          obs:  np.ndarray shape (obs_dim,)
          info: {}
        """
        super().reset(seed=seed)

        self._init_grid(seed)

        pos_config = np.array(self.start_pos)
        self.current_step = 0

        if pos_config.ndim == 1:
            self.agent_pos = pos_config.astype(int) + 1
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
                flat_idx = self.np_random.choice(probs.size, p=probs.ravel())
                rr, cc = np.unravel_index(flat_idx, probs.shape)
                self.agent_pos = np.array([rr + 1, cc + 1], dtype=int)
            else:
                self.agent_pos = np.array([1, 1], dtype=int)

        self.see_pos = self.agent_pos.copy()

        return self._get_obs(), {}

    def step(self, action):
        r"""Выполняет одно действие в среде.

        Награда:
          - self.step_reward за обычный шаг;
          - self.goal_reward при достижении цели.
        """

        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = int(action)
            elif action.ndim == 1 and action.size == 1:
                action = int(action[0])

        action = int(action)

        v = np.array(
            [
                [-1, 0],
                [0, 1],
                [1, 0],
                [0, -1],
            ],
            dtype=int,
        )

        move = v[action]

        self.current_step += 1

        target_pos = self.agent_pos + move
        see_value = self.grid[tuple(target_pos)]

        # базовая награда за шаг (может быть 0.0 или отрицательной)
        reward = self.step_reward
        terminated = False
        truncated = False

        is_blocked = (see_value == self.feature_wall) or (see_value == self.feature_obstacle)

        if is_blocked:
            if self.see_obstacle:
                # Агент видит препятствие/стену
                self.see_pos = target_pos
            else:
                # Агент не видит препятствие, видит только свою позицию
                self.see_pos = self.agent_pos
        else:
            self.agent_pos = target_pos
            self.see_pos = target_pos

            if see_value == self.feature_goal:
                reward = self.goal_reward
                terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {}

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

        palette_colors = [base_cmap(i % base_cmap.N) for i in range(self.n_colors)]
        palette_colors.append("black")       # Стена
        palette_colors.append("firebrick")   # Препятствие
        palette_colors.append("green")       # Цель

        my_cmap = colors.ListedColormap(palette_colors)
        max_val = self.feature_goal
        bounds = np.arange(-0.5, max_val + 1.5, 1)
        norm = colors.BoundaryNorm(bounds, my_cmap.N)

        ax.imshow(self.grid, cmap=my_cmap, norm=norm, origin='upper')

        ay, ax_coord = self.agent_pos
        ax.scatter(
            ax_coord,
            ay,
            c='gold',
            s=400,
            marker='*',
            label='Agent',
            edgecolors='black',
            zorder=10,
        )

        ax.set_xticks(np.arange(-0.5, self.w + 2, 1))
        ax.set_yticks(np.arange(-0.5, self.h + 2, 1))
        ax.grid(which='major', color='gray', linestyle='-', linewidth=2)
        ax.tick_params(
            axis='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

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