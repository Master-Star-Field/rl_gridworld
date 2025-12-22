import numpy as np
from typing import Tuple, Optional, List

def generate_walls(
    h: int,
    w: int,
    obstacle_ratio: float,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""# generate_obstacle_mask

    Генерирует булеву матрицу препятствий 'obstacle_mask' размера '(h, w)' c
    заданной долей 'True' (препятствий) и **гарантированным путём** между
    двумя точками при движении по 4‑связности (вверх/вниз/влево/вправо).

    Путь обеспечивается тем, что между 'start' и 'goal' строится
    простая монотонная "манхэттенская" траектория, и клетки этой
    траектории **никогда** не помечаются как препятствия.

    > Координаты 'start' и 'goal' задаются в системе внутренних клеток
    > маски: '(row, col)' в диапазоне '[0, h-1] × [0, w-1]'.

    Parameters

    h : int
        Высота маски (количество строк).
    w : int
        Ширина маски (количество столбцов).
    obstacle_ratio : float
        Желаемая доля препятствий (значений True) в [0.0, 1.0].
        Фактическая доля может быть чуть меньше, так как клетки пути
        не могут быть препятствиями.
    start : Tuple[int, int]
        Начальная точка пути (row, col).
    goal : Tuple[int, int]
        Конечная точка пути (row, col).
    rng : np.random.Generator, optional
        Генератор случайных чисел. Если None — создаётся новый.

    Returns

    np.ndarray
        Булева матрица формы '(h, w)', где 'True' — препятствие.
    """
    if rng is None:
        seed_seq = np.random.SeedSequence(
            [
                int(h),
                int(w),
                int(round(obstacle_ratio * 10_000)),
            ]
        )
        rng = np.random.default_rng(seed_seq)
    if not (0.0 <= obstacle_ratio <= 1.0):
        raise ValueError("obstacle_ratio должно быть в диапазоне [0.0, 1.0].")

    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < h and 0 <= sc < w):
        raise ValueError(f"start={start} вне границ массива {h}x{w}.")
    if not (0 <= gr < h and 0 <= gc < w):
        raise ValueError(f"goal={goal} вне границ массива {h}x{w}.")

    mask = np.zeros((h, w), dtype=bool)

    # манхеттенский путь
    path: List[Tuple[int, int]] = []
    r, c = sr, sc
    path.append((r, c))

    while (r, c) != (gr, gc):
        moves: List[Tuple[int, int]] = []

        # Двигаемся по строке
        if r < gr:
            moves.append((1, 0))
        elif r > gr:
            moves.append((-1, 0))

        # Двигаемся по столбцу
        if c < gc:
            moves.append((0, 1))
        elif c > gc:
            moves.append((0, -1))

        # Выбираем одно из доступных направлений случайно
        dr, dc = moves[rng.integers(len(moves))]
        r += dr
        c += dc
        path.append((r, c))

    path_set = set(path)

    total_cells = h * w
    desired_obstacles = int(round(obstacle_ratio * total_cells))

    all_cells = [(i, j) for i in range(h) for j in range(w)]
    candidate_cells = [cell for cell in all_cells if cell not in path_set]

    max_obstacles = len(candidate_cells)
    n_obstacles = min(desired_obstacles, max_obstacles)

    if n_obstacles > 0:
        chosen_idx = rng.choice(len(candidate_cells), size=n_obstacles, replace=False)
        for idx in chosen_idx:
            rr, cc = candidate_cells[idx]
            mask[rr, cc] = True

    return mask