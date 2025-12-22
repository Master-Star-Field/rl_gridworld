# `save_gif.py` — Документация и Заметки

---

## `save_episode_gif(env, model, ...) -> bool`

Записывает один эпизод взаимодействия агента со средой и сохраняет его как GIF-анимацию.

### Параметры:
- `env`: Среда OpenAI Gym-совместимая.
    - Поддерживаются:
        - Векторные среды (с `_render_map()`)
        - Простые среды с `render_map()` или `render()`
- `model`:
    - Сеть PyTorch (`torch.nn.Module`)
    - Или объект агента с:
        - `.act_greedy()`
        - `.net` и `.device`
- `filename` (str): Имя выходного `.gif`.  
  Пример: `"episode_success.gif"`
- `fps` (int): Частота кадров. Лучше низкая (2–5), так как RL-эпизоды медленные.
- `max_frames` (int): Максимальное число кадров, чтобы избежать зацикливания.
- `hold_last` (int): Сколько раз продублировать последний кадр. Помогает замедлить конец GIF'а визуально.

### Возвращает:
```python
True  # если был достигнут terminal (успех)
False # иначе — например, при обрезке из-за max_frames
```

### Логика работы

1. **Обнаружение типа модели**:
    ```python
    if hasattr(model, "act_greedy"): ...
    elif hasattr(model, "net"): ...
    else: считаем, что model — это чистая нейросеть
    ```

2. **Подготовка среды отрисовки**:
    - Переключаем backend `matplotlib` на `"Agg"` (без GUI).
    - Сохраняем старый backend для восстановления.

3. **Функция `render_frame()` — приватная**  
   Пробует несколько способов отрендерить текущее состояние:

   ??? example "Приоритеты отрисовки"
       1. `env._render_map(return_rgb_array=True)` — для векторных сред (batch-рендер)
       2. `env.render_map()` — специфичный метод GridWorld (рисует в plt)
       3. `env.render()` — стандартный Gym-рендер
          - Может возвращать `np.ndarray` (RGB)
          - Или рисовать на `matplotlib` — тогда захватываем буфер

   !!! note "Примечание"
       Используется `plt.gcf()` + `canvas.buffer_rgba()` для захвата изображения.

   !!! failure "Ошибка"
       Если ни один из методов не найден — `raise AttributeError`.

4. **Инициализация среды**
    ```python
    obs, _ = env.reset()
    ```
    - Поддержка как одиночной, так и векторной среды:
      - `obs.ndim == 1` → `n_envs = 1`
      - `obs.ndim == 2` → `n_envs = B` (batch size)

5. **Цикл агента**
    - Пока не завершены все среды (`not done_vec.all()`) и лимит кадров не превышен.

    **Два режима поведения**:

    ??? mode "Агент с `.act_greedy()`"
        ```python
        action, hidden = agent.act_greedy(obs, hidden=hidden)
        ```
        Применяется ко всем политическим агентам (PPO, A2C и т.д.), возвращающим действие напрямую.

    ??? mode "Сырая нейросеть"
        - Подготавливаем батч: `obs_batch = obs.reshape(1, -1)` или оставляем как есть.
        - Если модель рекуррентная (`hasattr(net, "init_hidden")`), инициализируем `hidden`.
        - Прогон через сеть:
          ```python
          with torch.no_grad():
              out = net(obs_t, hidden) if hidden else net(obs_t)
          ```
        - Выбор действия:
          $$
          \pi(a|s) = \arg\max_a \logits_a
          $$
        - Преобразуем в `numpy` и подаём в `env.step()`.

6. **Обновление состояния**
    - Получаем `next_obs`, `reward`, `terminated`, `truncated`.
    - Объединяем флаги:
      $$
      \text{done}_t = \text{terminated}_t \lor \text{truncated}_t
      $$
    - Обновляем `done_vec` по-векторно.
    - `terminated_flag = True`, если **хотя бы один** `terminated`.

7. **Сборка GIF**
    - Если был `terminated`, дублируем последний кадр `hold_last` раз:
      ```python
      for _ in range(hold_last - 1):
          frames.append(last_frame.copy())
      ```
      Это даёт эффект "паузы" в конце GIF.

    - Сохранение:
      ```python
      imageio.mimsave(filename, frames, fps=fps, loop=1)
      ```
      где `loop=1` — бесконечный цикл проигрывания.

8. **Очистка**
    - Всегда возвращаем исходный `matplotlib` backend.
    - Переводим сеть в режим обучения: `net.train()` — важно при использовании `BatchNorm` или `Dropout`.

### Пример вызова

```python
save_episode_gif(
    env=env,
    model=agent,
    filename="success_run.gif",
    fps=3,
    max_frames=150,
    hold_last=15
)
```

### Особенности

!!! warning "Внимание"
    Убедитесь, что CUDA-устройство доступно, если модель на `cuda`.

!!! tip "Совет"
    Используйте `hold_last=15` и `fps=2` для плавного завершения анимации.

### Возможные расширения

- Добавить поддержку `info` для аннотаций на GIF.
- Возможность передавать `seed` в `reset()`.
- Визуализация Q-значений поверх карты.

---
</details>
</file>
```