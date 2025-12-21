from typing import Optional, Union, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from torchvision import datasets

from .grid_world import GridWorldEnv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors


class GridWorldMnistEnv(GridWorldEnv):
    r"""
    Вариант GridWorld, где наблюдение кодируется случайным 
    изображением из MNIST по "цифре", соответствующей типу клетки.

    Соответствие следующее:
    
      - пол цвета c (0 .. n_colors-1) -> цифра c;
      - стена               -> цифра 7;
      - препятствие         -> цифра 8;
      - цель                -> цифра 9.

    Ограничения:
      - изображение вырезается как центр-кроп из 28x28 MNIST
        до размера image_size x image_size (image_size >= 12).

    Наблюдение:
      - если flatten=True:
            obs.shape = (image_size * image_size,)
      - иначе:
            obs.shape = (image_size, image_size)

    Параметр step_reward наследуется от GridWorldEnv:
      - по умолчанию 0.0;
      - можно задать отрицательное значение, чтобы ускорить достижение цели.
    """

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
        step_reward: float = 0.0,
        mnist_root: str = "./mnist_data",
        image_size: int = 14,
        flatten: bool = True,
        digit_render_mode: str = "text",
    ):
        if n_colors > 7:
            raise ValueError(
                f"GridWorldMnistEnv: n_colors={n_colors} > 7 не поддерживается "
                f"(максимум 7 цветов пола, +3 служебных)"
            )

        super().__init__(
            h=h,
            w=w,
            obstacle_mask=obstacle_mask,
            n_colors=n_colors,
            pos_goal=pos_goal,
            pos_agent=pos_agent,
            see_obstacle=see_obstacle,
            render_mode=render_mode,
            seed=seed,
            step_reward=step_reward,
        )

        self.image_size = int(image_size)
        self.flatten = bool(flatten)
        self.digit_render_mode = str(digit_render_mode)

        if self.image_size < 12 or self.image_size > 28:
            raise ValueError(
                f"image_size должен быть в диапазоне [12, 28], получено {self.image_size}"
            )

        mnist = datasets.MNIST(
            root=mnist_root,
            train=True,
            download=True,
        )
        self._mnist_images = mnist.data.numpy()
        self._mnist_labels = mnist.targets.numpy()

        self._digit_indices = {
            d: np.where(self._mnist_labels == d)[0]
            for d in range(10)
        }

        if self.flatten:
            obs_shape = (self.image_size * self.image_size,)
        else:
            obs_shape = (self.image_size, self.image_size)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32,
        )

        self._mnist_mosaic: Optional[np.ndarray] = None

    def _tile_type_to_digit(self, tile_val: int) -> int:
        """Преобразует тип клетки (0..n_colors+2) в цифру MNIST (0..9)."""
        if 0 <= tile_val < self.n_colors:
            return int(tile_val)
        if tile_val == self.feature_wall:
            return 7
        if tile_val == self.feature_obstacle:
            return 8
        if tile_val == self.feature_goal:
            return 9
        return 0

    def _sample_digit_image(self, digit: int) -> np.ndarray:
        """
        Выбирает случайное изображение цифры digit из MNIST,
        приводит к нужному размеру (центр‑кроп) и нормирует в [0,1].
        """
        idxs = self._digit_indices.get(digit, None)
        if idxs is None or len(idxs) == 0:
            raise RuntimeError(f"Нет ни одной картинки для цифры {digit} в MNIST.")

        idx = int(self.np_random.choice(idxs))
        img = self._mnist_images[idx]  # (28, 28), uint8

        if self.image_size < 28:
            start = (28 - self.image_size) // 2
            end = start + self.image_size
            img = img[start:end, start:end]

        img = img.astype(np.float32) / 255.0  # в [0, 1]
        return img

    def _build_mnist_mosaic(self) -> None:
        """
        Строит 2D-мозаику всего поля из MNIST-изображений
        (включая рамку стен) для режима digit_render_mode="mnist".
        """
        H = (self.h + 2) * self.image_size
        W = (self.w + 2) * self.image_size
        mosaic = np.zeros((H, W), dtype=np.float32)

        for gr in range(self.h + 2):
            for gc in range(self.w + 2):
                tile_val = int(self.grid[gr, gc])
                digit = self._tile_type_to_digit(tile_val)
                img = self._sample_digit_image(digit)
                r0 = gr * self.image_size
                c0 = gc * self.image_size
                mosaic[r0:r0 + self.image_size, c0:c0 + self.image_size] = img

        self._mnist_mosaic = mosaic

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Перезапуск среды.
        """
        obs, info = super().reset(seed=seed, options=options)
        self._mnist_mosaic = None
        return obs, info

    def _get_obs(self) -> np.ndarray:
        """
        Наблюдение: случайное MNIST-изображение по типу клетки в позиции see_pos.

        Возвращает:
          - если flatten=True: shape = (image_size * image_size,)
          - иначе:            shape = (image_size, image_size)
        """
        tile_val = self.grid[tuple(self.see_pos)]
        digit = self._tile_type_to_digit(int(tile_val))
        img = self._sample_digit_image(digit)

        if self.flatten:
            return img.reshape(-1).astype(np.float32)
        else:
            return img.astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            self.render_map()

    def render_map(self):
        """
        Визуализация поля.

        digit_render_mode:
          - "text" : цветные тайлы + напечатанные цифры;
          - "mnist": поле из MNIST-изображений (мозаика), агент поверх.
        """
        if self.digit_render_mode == "mnist":
            self._render_map_mnist()
        else:
            self._render_map_text()

    def _render_map_text(self):
        """Режим визуализации: цветные тайлы + напечатанные цифры."""
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

        for r in range(self.h):
            for c in range(self.w):
                gr = r + 1
                gc = c + 1
                val = int(self.grid[gr, gc])
                digit = self._tile_type_to_digit(val)

                bg = my_cmap(norm(val))
                brightness = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                text_color = "black" if brightness > 0.5 else "white"

                ax.text(
                    gc,
                    gr,
                    str(digit),
                    ha='center',
                    va='center',
                    color=text_color,
                    fontsize=10,
                    fontweight='bold',
                    zorder=11,
                )

        patches = [mpatches.Patch(color='gold', label='Agent')]
        for i in range(self.n_colors):
            patches.append(
                mpatches.Patch(
                    color=palette_colors[i],
                    label=f'Floor {i} (digit {i})'
                )
            )
        patches.append(mpatches.Patch(color='black', label='Wall (digit 7)'))
        patches.append(mpatches.Patch(color='firebrick', label='Obstacle (digit 8)'))
        patches.append(mpatches.Patch(color='limegreen', label='Goal (digit 9)'))

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"GridWorldMNIST (text) {self.h}x{self.w}")
        plt.tight_layout()

        import matplotlib as mpl
        if mpl.get_backend().lower() != "agg":
            plt.show()

    def _render_map_mnist(self):
        """Режим визуализации: поле из MNIST-изображений (мозаика), агент поверх."""
        if self._mnist_mosaic is None:
            self._build_mnist_mosaic()

        if plt.get_fignums():
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(9, 7))
        ax = plt.gca()

        ax.imshow(self._mnist_mosaic, cmap='gray', vmin=0.0, vmax=1.0, origin='upper')

        ay, ax_coord = self.agent_pos
        y_pix = ay * self.image_size + (self.image_size - 1) / 2.0
        x_pix = ax_coord * self.image_size + (self.image_size - 1) / 2.0

        ax.scatter(
            x_pix,
            y_pix,
            c='red',
            s=200,
            marker='*',
            label='Agent',
            edgecolors='white',
            linewidths=1.5,
            zorder=10,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"GridWorldMNIST (mnist) {self.h}x{self.w}")
        ax.legend(loc='upper right')

        plt.tight_layout()

        import matplotlib as mpl
        if mpl.get_backend().lower() != "agg":
            plt.show()