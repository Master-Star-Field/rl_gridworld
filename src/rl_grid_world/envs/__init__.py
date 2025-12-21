from gymnasium.envs.registration import register

from .grid_world import GridWorldEnv
from .mnist_grid_world import GridWorldMnistEnv
from .gym_grid_world import GymVectorGridWorldEnv
from .vector_grid_world import VectorGridWorldEnv

__all__ = [
    "GridWorldEnv",
    "GridWorldMnistEnv",
    "GymVectorGridWorldEnv",
    "VectorGridWorldEnv",
]

# Обычная one-hot среда
register(
    id="GridWorld-OneHot-v0",
    entry_point="src.rl_grid_world.envs.grid_world:GridWorldEnv",
)

# MNIST-вариант
register(
    id="GridWorld-MNIST-v0",
    entry_point="src.rl_grid_world.envs.mnist_grid_world:GridWorldMnistEnv",
    kwargs={
        "image_size": 14,
        "flatten": True,
        "digit_render_mode": "text",
    },
)

# Векторная среда на базе SyncVectorEnv
register(
    id="GridWorld-OneHot-VectorGym-v0",
    entry_point="src.rl_grid_world.envs.gym_grid_world:GymVectorGridWorldEnv",
    kwargs={
        "n_envs": 16,
    },
)

# Полностью numpy-векторизованная среда
register(
    id="GridWorld-OneHot-VectorNP-v0",
    entry_point="src.rl_grid_world.envs.vector_grid_world:VectorGridWorldEnv",
    kwargs={
        "n_envs": 16,
    },
)