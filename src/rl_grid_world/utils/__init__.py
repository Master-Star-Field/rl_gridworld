"""
Вспомогательные функции для GridWorld.

Содержит:
- generate_walls — генерация маски препятствий;
- save_episode_gif — запись GIF по одному эпизоду.
"""

from .generate_wall import generate_walls
from .save_gif import save_episode_gif

__all__ = [
    "generate_walls",
    "save_episode_gif",
]