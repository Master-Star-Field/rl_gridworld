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

Диаграмма состояний агента:

```mermaid
stateDiagram-v2
    [*] --> Активен
    Активен --> Заблокирован: Столкновение со стеной/препятствием
    Активен --> Цель_достигнута: Достижение клетки с целью
    Активен --> Активен: Успешное перемещение
    Заблокирован --> Активен: Продолжение действий
    Цель_достигнута --> [*]: Завершение эпизода
```

Note:
    - Среда использует одномерные one-hot векторы для наблюдений.
    - Координаты в сетке начинаются с (0, 0) в верхнем левом углу.
    - Использование `see_obstacle=True` позволяет получать информацию о препятствиях.
"""

from typing import Optional, Union, Tuple
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    
    """Среда-сетка для задач обучения с подкреплением.

    Среда представляет собой двумерную сетку размером h×w, окруженную стенами.
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

    Action Space:
        Пространство действий — дискретное с 4 направлениями:

        - 0: вверх     ( [-1,  0] )
        - 1: вправо    ( [ 0,  1] )
        - 2: вниз      ( [ 1,  0] )
        - 3: влево     ( [ 0, -1] )

        Действие применяется к текущей позиции агента:

        \[\text{target\_pos} = \text{agent\_pos} + \text{move\_vector}\]

    Observation Space:
        Пространство наблюдений — вещественный вектор размерности
        (n_colors + 3), закодированный как one-hot вектор,
        где 3 дополнительных признака соответствуют стене, препятствию и цели.

        \[\text{shape} = (\text{n\_colors} + 3,)\]

        Например, при n_colors=2:

        - [1, 0, 0, 0, 0] — пол типа 0
        - [0, 1, 0, 0, 0] — пол типа 1
        - [0, 0, 1, 0, 0] — стена
        - [0, 0, 0, 1, 0] — препятствие
        - [0, 0, 0, 0, 1] — цель

        Наблюдение зависит от see_pos, которая может отличаться от agent_pos,
        если агент столкнулся с препятствием.

    Reward:
        Награда +1.0 начисляется только при достижении цели (клетки GOAL).
        В остальных случаях награда равна 0.0.

    Episode Termination:
        Эпизод завершается, когда агент достигает клетки с целью.

    Rendering:
        Поддерживает режимы:

        - 'human': Отображение в окне с помощью matplotlib
        - 'rgb_array': Возврат RGB-массива изображения

    Example:
        >>> env = GridWorldEnv(h=4, w=4, n_colors=2, obstacle_mask=np.array([
        ...     [False, True,  False, False],
        ...     [False, False, False, False],
        ...     [True,  False, False, True],
        ...     [False, False, False, False]]), pos_goal=(3, 3))
        >>> obs, info = env.reset(seed=42)
        >>> print('Наблюдение:', obs)
        Наблюдение: [1. 0. 0. 0. 0.]
        >>> print('Позиция агента:', env.agent_pos)
        Позиция агента: [1 1]
        >>> action = 2  # вниз
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> print('Новое наблюдение:', obs)
        Новое наблюдение: [0. 1. 0. 0. 0.]
        >>> env.close()

    Note:
        Среда окружена стенами размером в один элемент по периметру,
        поэтому внутренние координаты смещены на (1, 1) относительно
        индексов массива self.grid.
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
        render_mode: Optional[str] = None
        ):
        """Инициализирует среду GridWorld.

        Args:
            h: Высота внутренней сетки. По умолчанию 10.
            w: Ширина внутренней сетки. По умолчанию 10.
            obstacle_mask: Булев массив (h, w), где True — препятствие. 
                           Если None, препятствий нет.
            n_colors: Количество типов пола (цветов). По умолчанию 1.
            pos_goal: Позиция цели в формате (row, col). По умолчанию (0, 0).
            pos_agent: Позиция агента.
                        Может быть:
                        - кортеж (r, c) — фиксированная позиция
                        - массив (h, w) — распределение вероятностей
                        По умолчанию (1, 1).
            see_obstacle: Если True, агент "видит" препятствия при столкновении
                          (see_pos обновляется). Иначе see_pos остаётся на позиции агента.
            render_mode: Режим отрисовки. Допустимые значения: 'human', 'rgb_array'.

        Raises:
            ValueError: Если pos_goal вне границ сетки.

        Example:
            >>> # Среда 5x5 с 2 цветами пола и 3 препятствиями
            >>> mask = np.array([
            ...     [0, 1, 0, 0, 0],
            ...     [0, 1, 0, 1, 0],
            ...     [0, 0, 0, 0, 0],
            ...     [0, 0, 0, 0, 0],
            ...     [0, 0, 0, 0, 0]
            ... ]).astype(bool)
            >>> env = GridWorldEnv(h=5, w=5, n_colors=2, obstacle_mask=mask,
            ...                    pos_goal=(4, 4), pos_agent=(1, 1),
            ...                    see_obstacle=True, render_mode='human')

        Note:
            Если obstacle_mask передан в транспонированном виде (w, h),
            он автоматически транспонируется для соответствия (h, w).
        """
        self.h = h
        self.w = w
        self.n_colors = n_colors
        self.pos_goal = np.array(pos_goal)
        self.start_pos = pos_agent
        self.see_obstacle = see_obstacle
        self.render_mode = render_mode

        if np.any(self.pos_goal < 0) or np.any(self.pos_goal >= np.array([h, w])):
            raise ValueError(f"pos_goal {self.pos_goal} вне границ сетки {h}x{w}")

        if obstacle_mask is None:
            self.obstacle_mask = np.zeros((h, w), dtype=bool)
        else:
            self.obstacle_mask = np.array(obstacle_mask, dtype=bool)
            if self.obstacle_mask.shape == (w, h) and w != h:
                self.obstacle_mask = self.obstacle_mask.T
        
        self.feature_wall = n_colors
        self.feature_obstacle = n_colors + 1
        self.feature_goal = n_colors + 2
            
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.feature_goal + 1,), 
            dtype=np.float32
        )
        
        # Сетка с дополнительными стенами по краям: (h+2, w+2)
        self.grid = np.zeros((h + 2, w + 2), dtype=int)
        
        self.agent_pos = np.array([0, 0])
        self.see_pos = np.array([0, 0])
        
    def _init_grid(self, seed: Optional[int]):
        
        """Инициализирует внутреннюю сетку среды.

        Заполняет сетку следующим образом:
        1. Вся сетка заполняется стенами (значение = feature_wall)
        2. Внутренняя область (h×w) заполняется случайными типами пола
           или препятствиями в соответствии с obstacle_mask
        3. Клетка цели устанавливается в заданной позиции

        Args:
            seed: Сид для генератора случайных чисел. Используется для
                  инициализации типов пола.

        Process:
            
            ```mermaid
            graph TD
                A[Начало] --> B[Заполнить стенами]
                B --> C[Сгенерировать случайные типы пола]
                C --> D[Применить obstacle_mask]
                D --> E[Установить позицию цели]
                E --> F[Завершено]
            ```

        Example:
            >>> env = GridWorldEnv(h=3, w=3, n_colors=2, obstacle_mask=np.array([
            ...     [0, 1, 0],
            ...     [0, 0, 0],
            ...     [1, 0, 0]
            ... ]).astype(bool), pos_goal=(2, 2))
            >>> env._init_grid(seed=42)
            >>> print(env.grid)  
            [[3 3 3 3 3]
             [3 0 2 1 3]
             [3 1 1 1 3]
             [3 2 0 2 3]
             [3 3 3 3 3]]
        """
        rng = np.random.default_rng(seed)
        
        self.grid.fill(self.feature_wall)
        
        
        random_floors = rng.integers(0, self.n_colors, size=(self.h, self.w))
        
        inner_area = np.where(
            self.obstacle_mask,
            self.feature_obstacle,
            random_floors
        )
        

        self.grid[1:-1, 1:-1] = inner_area
        
        self.grid[tuple(self.pos_goal + 1)] = self.feature_goal

    def _get_obs(self):
        
        """Возвращает наблюдение агента в виде one-hot вектора.

        Наблюдение основано на позиции self.see_pos, которая может отличаться
        от позиции агента, если произошло столкновение с препятствием.

        Returns:
            np.ndarray: One-hot вектор формы (n_colors + 3,) типа float32,
                        где индекс значения 1.0 соответствует типу клетки.

        Formula:
            \[\text{obs}[i] = \begin{cases}
            1, & \text{если } i = \text{grid}[\text{see\_pos}] \\
            0, & \text{иначе}
            \end{cases}\]

        Example:
            >>> env = GridWorldEnv(n_colors=2)
            >>> env._init_grid(seed=1)
            >>> env.agent_pos = np.array([1, 1])
            >>> env.see_pos = np.array([1, 1])  # Пол типа 0
            >>> obs = env._get_obs()
            >>> print(obs)
            [1. 0. 0. 0. 0.]
            >>> obs_idx = np.argmax(obs)
            >>> print(f'Тип клетки: {obs_idx}')
            Тип клетки: 0
        """
        
        obs_vec = np.zeros((self.feature_goal + 1,), dtype=np.float32)
        val = self.grid[tuple(self.see_pos)]
        obs_vec[val] = 1.0
        return obs_vec

    def reset(self, seed=None, options=None):
        """Перезапускает среду в начальное состояние.

        Args:
            seed: Сид для генерации случайных чисел.
            options: Дополнительные параметры сброса (не используются).

        Returns:
            Tuple[np.ndarray, dict]:
                - Наблюдение после сброса
                - Словарь дополнительной информации (пустой)

        Process:
            1. Инициализация генератора случайных чисел
            2. Создание новой сетки с помощью _init_grid
            3. Установка начальной позиции агента:
               - Если start_pos — вектор (напр. [1, 1]), используем напрямую
               - Если start_pos — матрица (h, w), выбираем позицию
                 с вероятностями из этой матрицы
            4. see_pos устанавливается равным agent_pos

        Raises:
            ValueError: Если матрица start_pos несовместима с размерами сетки.

        Example:
            # Фиксированная начальная позиция
            >>> env = GridWorldEnv(pos_agent=(2, 2))
            >>> obs, info = env.reset(seed=42)
            >>> print(env.agent_pos)
            [2 2]

            # Случайная начальная позиция по распределению
            >>> start_probs = np.ones((5, 5))
            >>> start_probs[0, :] = 0  # Запрет первой строки
            >>> start_probs[:, 0] = 0  # Запрет первого столбца
            >>> env = GridWorldEnv(h=5, w=5, pos_agent=start_probs)
            >>> obs, info = env.reset(seed=42)
            >>> print(env.agent_pos)  # Будет в диапазоне [1:4, 1:4]
            [3 2]
        """
        super().reset(seed=seed)
        self._init_grid(seed)
        
        pos_config = np.array(self.start_pos)
        
        if pos_config.ndim == 1:
            self.agent_pos = pos_config
        else:
            probs = np.array(pos_config, dtype=np.float32)
            
            if probs.shape == (self.w, self.h) and self.w != self.h:
                 probs = probs.T
                 
            probs[self.obstacle_mask] = 0
            probs[tuple(self.pos_goal)] = 0
            
            total = probs.sum()
            if total > 0:
                probs /= total  
                flat_idx = self.np_random.choice(probs.size, p=probs.flatten())
                coords = np.unravel_index(flat_idx, probs.shape)
                self.agent_pos = np.array(coords)
            else:
                self.agent_pos = np.array([1, 1])

        self.see_pos = self.agent_pos.copy()
        
        return self._get_obs(), {}

    def step(self, action: int):
        """Выполняет одно действие в среде.

        Args:
            action (int): Действие агента (0-3).

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                - Наблюдение после действия
                - Награда (1.0 при достижении цели, иначе 0.0)
                - terminated: True, если агент достиг цели
                - truncated: False (не используется)
                - info: Дополнительная информация (пустой словарь)

        Process:
            
            ```mermaid
            graph TD
                A[Начало] --> B[Вычислить целевую позицию]
                B --> C{Проверка проходимости}
                C -->|Заблокировано| D[Обновить see_pos]
                C -->|Свободно| E[Обновить agent_pos и see_pos]
                E --> F{Цель достигнута?}
                F -->|Да| G[Установить terminated=True, reward=1.0]
                F -->|Нет| H[reward=0.0]
                G --> I[Возврат результата]
                H --> I
                D --> I
            ```

        Formula:
            \[\text{target\_pos} = \text{agent\_pos} + \vec{v}[\text{action}]\]

            где \(\vec{v} = [ [-1,0], [0,1], [1,0], [0,-1] ]\)

        Example:
            >>> env = GridWorldEnv(h=3, w=3, pos_goal=(2, 2))
            >>> obs, _ = env.reset(seed=1)
            >>> print('До действия:', env.agent_pos)
            До действия: [1 1]
            >>> obs, reward, terminated, truncated, info = env.step(2)  # вниз
            >>> print('После действия:', env.agent_pos)
            После действия: [2 1]
            >>> obs, reward, terminated, truncated, info = env.step(1)  # вправо
            >>> print('Награда:', reward)
            Награда: 1.0
            >>> print('Завершено:', terminated)
            Завершено: True
        """
        # Векторы движения: [вверх, вправо, вниз, влево]
        v = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        
        move = v[action]
        target_pos = self.agent_pos + move
        
        see_value = self.grid[tuple(target_pos)]
        
        reward = 0.0
        terminated = False
        

        is_blocked = (see_value == self.feature_wall) or (see_value == self.feature_obstacle)
        
        if is_blocked:
            if self.see_obstacle:
                # Агент "видит" препятствие
                self.see_pos = target_pos
            else:
                # Агент не видит препятствие, видит только свою позицию
                self.see_pos = self.agent_pos
        else:
            self.agent_pos = target_pos
            self.see_pos = target_pos 
            
            if see_value == self.feature_goal:
                reward = 1.0
                terminated = True
                
        return self._get_obs(), reward, terminated, False, {}
    
    def render(self):
        
        """Отображает текущее состояние среды.

        Использует matplotlib для визуализации сетки, агента, препятствий и цели.
        Поддерживает режим 'human' для отображения в окне.

        Note:
            Для режима 'rgb_array' необходимо реализовать дополнительную логику
            получения массива пикселей.

        Example:
            >>> env = GridWorldEnv(render_mode='human')
            >>> obs, _ = env.reset()
            >>> env.render()  # Покажет окно с сеткой
            >>> env.close()
        """
        if self.render_mode == "human":
            self.render_map()

    def render_map(self):
        
        """Визуализирует сетку с помощью matplotlib.

        Отображает:
        - Сетку с разными типами клеток
        - Позицию агента (золотая звезда)
        - Легенду с цветами

        Color Palette:
            - Пол (0 до n_colors-1): Цвета из colormap 'Set3'
            - Стена: чёрный
            - Препятствие: красный ('firebrick')
            - Цель: зелёный ('green')

        Grid Lines:
            Проводятся по границам клеток для лучшей видимости.

        Example:
            >>> env = GridWorldEnv(n_colors=3, render_mode='human')
            >>> env.reset(seed=1)
            >>> env.render_map()  # Откроется окно с визуализацией
        """
        
        if plt.get_fignums():
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(9, 7))
        
        ax = plt.gca()

        base_cmap = plt.get_cmap('Set3')
        
        palette_colors = []
        for i in range(self.n_colors):
            palette_colors.append(base_cmap(i % 12))
            
        palette_colors.append("black")       # Стена
        palette_colors.append("firebrick")   # Препятствие
        palette_colors.append("green")       # Цель
        
        my_cmap = colors.ListedColormap(palette_colors)
        
        max_val = self.feature_goal
        bounds = np.arange(-0.5, max_val + 1.5, 1)
        norm = colors.BoundaryNorm(bounds, my_cmap.N)

        ax.imshow(self.grid, cmap=my_cmap, norm=norm, origin='upper')

        ay, ax_coord = self.agent_pos
        ax.scatter(ax_coord, ay, c='gold', s=400, marker='*', 
                   label='Agent', edgecolors='black', zorder=10)

        ax.set_xticks(np.arange(-0.5, self.w + 2, 1))
        ax.set_yticks(np.arange(-0.5, self.h + 2, 1))
        ax.grid(which='major', color='gray', linestyle='-', linewidth=2)
        
        ax.tick_params(axis='both', bottom=False, left=False, 
                       labelbottom=False, labelleft=False)

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