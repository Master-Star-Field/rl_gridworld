import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import imageio
import re
import os
from typing import Optional, Tuple, Any


def sanitize_filename(filename: str) -> str:
    """
    Очищает имя файла от недопустимых символов для Windows/Linux.
    
    Заменяет:
      - * на x
      - пробелы на _
      - запятые на _
      - другие запрещённые символы на _
    """
    # Разделяем путь и имя файла
    dir_path = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    
    # Запрещённые символы в Windows: \ / : * ? " < > |
    # Заменяем * на x (для размеров типа 5x5)
    base_name = base_name.replace('*', 'x')
    
    # Заменяем пробелы и запятые на подчёркивания
    base_name = base_name.replace(' ', '_')
    base_name = base_name.replace(',', '_')
    
    # Убираем остальные запрещённые символы
    base_name = re.sub(r'[\\/:?"<>|]', '_', base_name)
    
    # Убираем множественные подчёркивания
    base_name = re.sub(r'_+', '_', base_name)
    
    # Убираем подчёркивания в начале и конце (но перед расширением)
    name, ext = os.path.splitext(base_name)
    name = name.strip('_')
    base_name = name + ext
    
    # Собираем обратно путь
    if dir_path:
        return os.path.join(dir_path, base_name)
    return base_name


def save_episode_gif(
    env,
    model,
    filename: str = "episode.gif",
    fps: int = 2,
    max_frames: int = 200,
    hold_last: int = 10,
) -> bool:
    """
    Сохраняет GIF одного эпизода, визуализируя поведение агента в среде 'env' 
    с использованием политики из 'model'.

    Функция пошагово выполняет эпизод, делая шаги в среде на основе действий, 
    предсказанных моделью, и сохраняет каждый кадр в анимированное GIF-изображение. 
    Поддерживаются как обычные, так и векторизованные среды (несколько параллельных сред), 
    а также модели с рекуррентными архитектурами (например, LSTMs/GRUs).

    Поддерживаемые типы моделей:
    - Модели с методом 'act_greedy' (например, A2CAgent, DRQNLightning).
    - Модели с атрибутом 'net' (например, 'DQNAgent').
    - "Чистые" PyTorch-модели, выводящие логиты действий.

    Поддерживаемые типы сред:
    - 'GridWorldEnv' с методом 'render_map()'.
    - Векторизованные среды с '_render_map(return_rgb_array=True)'.
    - Любые среды, реализующие 'render()' и возвращающие массив изображения.

    Аргументы:
        env: Среда Gym/Gymnasium. Должна поддерживать 'reset()' и 'step()'.
        model: Модель или агент, способный предсказывать действия.
        filename (str): Имя файла для сохранения GIF (по умолчанию "episode.gif").
        fps (int): Частота кадров в GIF (по умолчанию 2 кадра в секунду).
        max_frames (int): Максимальное количество кадров в эпизоде.
        hold_last (int): Сколько раз продублировать последний кадр при успехе.

    Возвращает:
        bool: 'True', если хотя бы одно состояние 'terminated=True' было встретено.
              'False', если эпизод завершился по 'truncated' или 'max_frames'.

    Исключения:
        RuntimeError: Если не удалось создать ни одного кадра.
        AttributeError: Если среда не поддерживает ни один из методов визуализации.
        ValueError: Если размерности наблюдений не соответствуют ожидаемым.
    """
    
    # ======= ИСПРАВЛЕНИЕ: Санитизируем имя файла =======
    original_filename = filename
    filename = sanitize_filename(filename)
    
    if filename != original_filename:
        print(f"[GIF] Имя файла исправлено: '{original_filename}' -> '{filename}'")
    
    # Создаём директорию если нужно
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    # ===================================================
    
    print(f"[GIF] Начинаем запись GIF в файл: {filename}...")

    prev_backend = matplotlib.get_backend()
    plt.switch_backend("Agg")

    frames = []
    use_agent = False
    agent = None
    net = None
    device = torch.device("cpu")

    # ======= Определяем тип модели и извлекаем сеть/устройство =======
    if hasattr(model, "act_greedy"):
        agent = model
        use_agent = True
        
        # Определяем устройство
        if hasattr(model, "device"):
            device = model.device
        elif hasattr(model, "net"):
            try:
                device = next(model.net.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        elif hasattr(model, "q_net"):
            try:
                device = next(model.q_net.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        
        # Для логирования определяем тип агента
        agent_type = type(model).__name__
        print(f"[GIF] Используем агента типа: {agent_type} с методом act_greedy")
        
        # Получаем сеть для переключения в eval режим
        if hasattr(model, "net"):
            net = model.net
        elif hasattr(model, "q_net"):
            net = model.q_net
            
    elif hasattr(model, "net"):
        agent = model
        net = agent.net
        device = getattr(agent, "device", torch.device("cpu"))
        print(f"[GIF] Используем модель с атрибутом .net")
    else:
        agent = None
        net = model
        try:
            device = next(net.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        print(f"[GIF] Используем чистую PyTorch модель")

    # Переключаем сеть в режим оценки
    if net is not None:
        net.eval()

    def render_frame() -> None:
        """Рендерит текущий кадр и добавляет в список frames."""
        if hasattr(env, "_render_map"):
            img = env._render_map(return_rgb_array=True)
            if img is not None:
                frames.append(img)
            return

        if hasattr(env, "render_map"):
            env.render_map()
            fig = plt.gcf()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image = np.asarray(buf, dtype=np.uint8)[..., :3]
            frames.append(image)
            plt.close(fig)
            return

        if hasattr(env, "render"):
            out = env.render()
            if isinstance(out, np.ndarray):
                frames.append(out)
            else:
                fig = plt.gcf()
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                image = np.asarray(buf, dtype=np.uint8)[..., :3]
                frames.append(image)
                plt.close(fig)
            return

        raise AttributeError(
            "Среда не поддерживает ни _render_map(), ни render_map(), ни render()."
        )

    def get_hidden_batch_size(hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> int:
        """Возвращает batch_size из hidden state или 0 если hidden is None."""
        if hidden is None:
            return 0
        h, c = hidden
        return h.shape[1]  # (num_layers, batch, hidden_dim)

    try:
        full_obs, _ = env.reset()
        obs = np.asarray(full_obs, dtype=np.float32)

        # Определяем количество под-сред
        if obs.ndim == 1:
            n_envs = 1
            obs_dim = obs.shape[0]
        elif obs.ndim == 2:
            n_envs = obs.shape[0]
            obs_dim = obs.shape[1]
        else:
            raise ValueError(
                f"[GIF] Ожидалось obs.ndim 1 или 2, получили {obs.ndim}"
            )

        print(f"[GIF] Среда: n_envs={n_envs}, obs_dim={obs_dim}")

        done_vec = np.zeros(n_envs, dtype=bool)
        terminated_flag = False

        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        frame_count = 0

        # Рендерим начальное состояние
        render_frame()
        frame_count += 1

        while (not done_vec.all()) and frame_count < max_frames:
            
            if use_agent:
                # ======= Используем act_greedy агента =======
                # Проверяем и сбрасываем hidden если нужно
                if hidden is not None:
                    hidden_batch = get_hidden_batch_size(hidden)
                    # Для A2C: batch_size должен соответствовать n_envs или 1
                    # Для DRQN: всегда batch_size=1
                    expected_batch = n_envs if obs.ndim == 2 else 1
                    
                    # Для DRQN act_greedy всегда работает с batch=1
                    if hasattr(agent, "q_net"):
                        expected_batch = 1
                    
                    if hidden_batch != expected_batch:
                        print(f"[GIF] Сбрасываем hidden: batch {hidden_batch} -> {expected_batch}")
                        hidden = None
                
                try:
                    action, hidden = agent.act_greedy(obs, hidden=hidden, env=env)
                except TypeError:
                    # act_greedy не принимает env
                    action, hidden = agent.act_greedy(obs, hidden=hidden)
                    
            else:
                # ======= Используем сеть напрямую =======
                obs_np = np.asarray(obs, dtype=np.float32)
                if obs_np.ndim == 1:
                    B = 1
                    obs_batch = obs_np.reshape(1, -1)
                elif obs_np.ndim == 2:
                    B = obs_np.shape[0]
                    obs_batch = obs_np
                else:
                    raise ValueError(
                        f"[GIF] Ожидалось obs.ndim 1 или 2, получили {obs_np.ndim}"
                    )

                # Инициализируем или сбрасываем hidden state
                if hasattr(net, "init_hidden"):
                    if hidden is None:
                        hidden = net.init_hidden(batch_size=B, device=device)
                    elif get_hidden_batch_size(hidden) != B:
                        print(f"[GIF] Re-init hidden: batch {get_hidden_batch_size(hidden)} -> {B}")
                        hidden = net.init_hidden(batch_size=B, device=device)
                elif hasattr(net, "get_initial_state"):
                    # Для DRQN
                    if hidden is None:
                        hidden = net.get_initial_state(batch_size=B, device=device)
                    elif get_hidden_batch_size(hidden) != B:
                        hidden = net.get_initial_state(batch_size=B, device=device)

                obs_t = torch.from_numpy(obs_batch).float().to(device)

                with torch.no_grad():
                    if hidden is not None:
                        out = net(obs_t, hidden)
                    else:
                        out = net(obs_t)

                    # Разбираем выход сети
                    if isinstance(out, tuple) and len(out) == 3:
                        logits, value, hidden = out
                    elif isinstance(out, tuple) and len(out) == 2:
                        logits, hidden = out
                    else:
                        logits = out
                        hidden = None

                    # Выбираем действие жадно
                    actions_tensor = torch.argmax(logits, dim=-1)  # (B,) или (B, T)
                    
                    # Если есть временное измерение, берём последний шаг
                    if actions_tensor.dim() > 1:
                        actions_tensor = actions_tensor[:, -1]

                actions_np = actions_tensor.cpu().numpy().astype(int)
                
                if B == 1:
                    action = int(actions_np[0])
                else:
                    action = actions_np

            # ======= Выполняем шаг в среде =======
            full_next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.asarray(full_next_obs, dtype=np.float32)

            # Обрабатываем done флаги
            term = np.atleast_1d(np.asarray(terminated, dtype=bool))
            trunc = np.atleast_1d(np.asarray(truncated, dtype=bool))

            # Проверяем размерность
            if term.shape[0] != n_envs:
                if term.shape[0] == 1 and n_envs == 1:
                    pass  # OK
                else:
                    # Если среда возвращает скаляр для векторной среды - расширяем
                    if term.shape[0] == 1:
                        term = np.full(n_envs, term[0], dtype=bool)
                        trunc = np.full(n_envs, trunc[0], dtype=bool)
                    else:
                        raise ValueError(
                            f"[GIF] Несоответствие размерности: n_envs={n_envs}, "
                            f"terminated.shape={term.shape}"
                        )

            step_done = np.logical_or(term, trunc)
            done_vec = np.logical_or(done_vec, step_done)
            terminated_flag = terminated_flag or bool(term.any())

            # Рендерим кадр
            render_frame()
            frame_count += 1

            if frame_count % 20 == 0:
                print(f"[GIF] Обработано кадров: {frame_count}, done: {done_vec.sum()}/{n_envs}")

        # ======= Финализация GIF =======
        print(f"[GIF] Эпизод завершён: frames={frame_count}, terminated={terminated_flag}")

        # Дублируем последний кадр для лучшей визуализации успеха
        if terminated_flag and len(frames) > 0 and hold_last > 1:
            last_frame = frames[-1]
            for _ in range(hold_last - 1):
                frames.append(last_frame.copy())

        if len(frames) > 0:
            imageio.mimsave(filename, frames, fps=fps, loop=0)
            print(f"[GIF] Успешно сохранено: {filename} ({len(frames)} кадров)")
            return bool(terminated_flag)
        else:
            print("[GIF] Ошибка: не удалось записать ни одного кадра.")
            raise RuntimeError("save_episode_gif: список frames пуст.")

    except Exception as e:
        print(f"[GIF] Критическая ошибка при записи GIF: {e}")
        raise

    finally:
        plt.switch_backend(prev_backend)
        if net is not None:
            net.train()