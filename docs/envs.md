# Среды (Environments)

Этот раздел описывает реализацию клеточных сред для обучения
с подкреплением в пакете `rl_grid_world.envs`.


## GridWorldEnv

::: rl_grid_world.envs.grid_world
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_signature: true
      show_source: false
      docstring_style: google
      members_order: source
      separate_signature: true



## Как использовать среду

Пример:

```python
from rl_grid_world.envs.grid_world import GridWorldEnv

env = GridWorldEnv(h=5, w=5, n_colors=5, render_mode="human")
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()