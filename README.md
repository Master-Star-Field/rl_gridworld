# GridWorld Environment

Среда GridWorld для задач обучения с подкреплением, реализованная на основе Gymnasium.

## Установка

Для установки среды и всех зависимостей выполните следующие команды:

```shell
cd rl-grid-world  # Перейдите в корень проекта
pip install -e .
```



## Примеры использования

### Базовый пример

```python
import gymnasium as gym
from rl_grid_world.envs import GridWorldEnv

# Создание среды
env = GridWorldEnv(
    h=5,
    w=5,
    n_colors=2,
    pos_goal=(4, 4),
    pos_agent=(1, 1),
    see_obstacle=True,
    render_mode="human"
)


obs, info = env.reset(seed=42)

for _ in range(20):
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
    
    env.render()

env.close()
```

### Пример с препятствиями

```python
import numpy as np

obstacle_mask = np.array([
    [False, True,  False, False, False],
    [False, True,  False, True,  False],
    [False, False, False, True,  False],
    [True,  False, False, False, False],
    [False, False, False, False, False]
])

env = GridWorldEnv(
    h=5,
    w=5,
    obstacle_mask=obstacle_mask,
    pos_goal=(4, 4),
    pos_agent=(1, 1),
    render_mode="human"
)

obs, info = env.reset(seed=42)
```

### Пример со случайной начальной позицией

```python
start_probs = np.ones((5, 5))
start_probs[0, :] = 0  
start_probs[:, 0] = 0  

env = GridWorldEnv(
    h=5,
    w=5,
    pos_agent=start_probs  
)

obs, info = env.reset(seed=42)  
print(f"Начальная позиция: {env.agent_pos}")
```

## Запуск примеров

Примеры скрипта доступен в файле /test.py:

```shell
python test.py
```

## Документация

Доступна по ссылке прикрепленной к репозиторию.

