from typing import Callable, List, Optional, Union, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv

from .grid_world import GridWorldEnv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors


class GymVectorGridWorldEnv(gym.Env):
    r"""
    Векторная GridWorld‑среда на базе gymnasium.vector.SyncVectorEnv.

    Внутри:
      - n_envs независимых копий одиночной GridWorldEnv;
      - каждую копию создаём отдельной фабрикой env_fn;
      - SyncVectorEnv управляет всем батчем.

    ВАЖНО:
      - Все под‑среды используют ОДИН И ТОТ ЖЕ seed для генерации карты
        (пол/препятствия), чтобы карта была идентичной;
      - Карта генерируется ОДИН РАЗ при создании среды и кэшируется;
      - при reset() используется закэшированная карта, гарантируя
        идентичность карт во всех эпизодах и под-средах.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        n_envs: int,
        make_env_kwargs: Optional[dict] = None,
        base_seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            n_envs: количество копий среды.
            make_env_kwargs: kwargs, которые будут переданы в GridWorldEnv(...)
                             для создания каждой копии среды.
            base_seed: базовый сид. Используется для генерации КАРТЫ (map_seed).
            render_mode: режим рендера для под‑сред ('human' или 'rgb_array').
        """

        super().__init__()

        if n_envs <= 0:
            raise ValueError("n_envs must be >= 1")
        self.n_envs = int(n_envs)

        if make_env_kwargs is None:
            make_env_kwargs = {}

        self._make_env_kwargs = dict(make_env_kwargs)
        self._make_env_kwargs.pop("seed", None)

        self._base_seed = base_seed
        self._render_mode = render_mode

        # ======= ИСПРАВЛЕНИЕ: фиксируем seed карты =======
        # Если base_seed не передан, генерируем случайный seed ОДИН РАЗ
        if base_seed is None:
            self._map_seed = np.random.randint(0, 2**31 - 1)
        else:
            self._map_seed = int(base_seed)
        # ==================================================

        env_fns: List[Callable[[], gym.Env]] = []

        for idx in range(self.n_envs):
            def make_env(i: int = idx) -> Callable[[], gym.Env]:
                def _thunk():
                    env = GridWorldEnv(
                        **self._make_env_kwargs,
                        seed=self._map_seed,
                        render_mode=render_mode,
                    )
                    return env
                return _thunk

            env_fns.append(make_env(idx))

        self.vec_env: SyncVectorEnv = SyncVectorEnv(env_fns)

        self.observation_space = self.vec_env.observation_space
        self.single_observation_space = self.vec_env.single_observation_space
        self.single_obs_dim = int(self.single_observation_space.shape[0])

        self.action_space: spaces.Discrete = self.vec_env.single_action_space

        example_env: GridWorldEnv = self.vec_env.envs[0]
        self.h = example_env.h
        self.w = example_env.w
        self.n_colors = example_env.n_colors

        # ======= ИСПРАВЛЕНИЕ: генерируем и кэшируем карту ОДИН РАЗ =======
        # Делаем начальный reset чтобы инициализировать карту
        self.vec_env.reset(seed=self._map_seed)
        
        # Кэшируем карту первой подсреды как эталон
        first_env: GridWorldEnv = self.vec_env.envs[0]
        self._cached_grid = first_env.grid.copy()
        
        # Сохраняем параметры для восстановления цели
        self._feature_goal = first_env.feature_goal
        self._pos_goal = first_env.pos_goal.copy()
        
        # Синхронизируем карты во всех подсредах
        self._sync_grids_from_cache()
        # ==================================================================

    def _sync_grids_from_cache(self) -> None:
        """
        Копирует закэшированную карту во все подсреды.
        Гарантирует идентичность карт (цвета пола, препятствия, цель).
        """
        for env in self.vec_env.envs:
            env.grid = self._cached_grid.copy()

    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """
        Сбрасывает ВСЕ под‑среды.
        
        ВАЖНО: карта (цвета пола, препятствия) НЕ перегенерируется при reset().
        Используется закэшированная карта, созданная при инициализации среды.
        Это гарантирует, что все эпизоды проходят на идентичной карте.
        
        seed:
          - int → используется для стартовых позиций агентов;
          - sequence[int] длины n_envs → отдельный сид для каждой среды.
        """
        obs, info = self.vec_env.reset(seed=seed, options=options)
        
        # ======= ИСПРАВЛЕНИЕ: восстанавливаем закэшированную карту =======
        # После reset() подсреды могли перегенерировать карту,
        # поэтому принудительно восстанавливаем закэшированную версию
        self._sync_grids_from_cache()
        
        # Обновляем наблюдения, т.к. карта могла измениться
        obs = np.array([env._get_obs() for env in self.vec_env.envs], dtype=np.float32)
        # ==================================================================
        
        return obs, info

    def step(self, action):
        """
        Векторный шаг:
          - action: int → одно действие для всех под‑сред;
          - action: np.ndarray shape (n_envs,) → своё действие для каждой.
        """
        if isinstance(action, np.ndarray):
            actions_arr = action.astype(int)
            if actions_arr.ndim == 0:
                actions_arr = np.full(self.n_envs, int(actions_arr), dtype=int)
            elif actions_arr.shape != (self.n_envs,):
                raise ValueError(
                    f"Ожидалось действие shape ({self.n_envs},) или скаляр, "
                    f"получено {actions_arr.shape}"
                )
        else:
            actions_arr = np.full(self.n_envs, int(action), dtype=int)

        obs, reward, terminated, truncated, info = self.vec_env.step(actions_arr)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.vec_env.close()

    def render(self):
        """
        Тип рендеринга берётся из исходной среды.
        """
        return self.vec_env.render()

    def _render_map(self, return_rgb_array: bool = False):
        """
        Рисует все под‑среды в сетке сабплотов.
        """

        envs = self.vec_env.envs
        n = self.n_envs

        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
        )
        axes = np.atleast_1d(axes).ravel()

        base_cmap = plt.get_cmap("Set3")
        palette_colors = [base_cmap(i % base_cmap.N) for i in range(self.n_colors)]
        palette_colors.append("black")      # стена
        palette_colors.append("firebrick")  # препятствие
        palette_colors.append("green")      # цель

        my_cmap = colors.ListedColormap(palette_colors)
        max_feature = self.n_colors + 2
        bounds = np.arange(-0.5, max_feature + 1.5, 1)
        norm = colors.BoundaryNorm(bounds, my_cmap.N)

        patches = [mpatches.Patch(color="gold", label="Agent")]
        for i in range(self.n_colors):
            patches.append(mpatches.Patch(color=palette_colors[i], label=f"Floor {i}"))
        patches.append(mpatches.Patch(color="black", label="Wall"))
        patches.append(mpatches.Patch(color="firebrick", label="Obstacle"))
        patches.append(mpatches.Patch(color="green", label="Goal"))

        for i in range(n):
            ax = axes[i]
            gw: GridWorldEnv = envs[i]

            ax.imshow(gw.grid, cmap=my_cmap, norm=norm, origin="upper")

            ay, ax_coord = gw.agent_pos
            ax.scatter(
                ax_coord,
                ay,
                c="gold",
                s=200,
                marker="*",
                edgecolors="black",
                zorder=10,
            )

            ax.set_xticks(np.arange(-0.5, gw.w + 2, 1))
            ax.set_yticks(np.arange(-0.5, gw.h + 2, 1))
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
            handles=patches,
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
            return None