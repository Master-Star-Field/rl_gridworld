import imageio
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt


def save_episode_gif(
    env,
    model,
    filename: str = "episode.gif",
    fps: int = 2,
    max_frames: int = 200,
    hold_last: int = 10,
):
    """
    Сохраняет GIF одного эпизода без открытия интерактивного окна.

    Параметры
    ---------
    env : GridWorldEnv
        Среда, в которой будет проигран эпизод.
    model : LightningModule с атрибутом `q_net`
        Обученный DRQN‑агент.
    filename : str
        Имя файла для сохранения GIF.
    fps : int
        Частота кадров.
    max_frames : int
        Максимальное количество шагов в эпизоде.
    hold_last : int
        Сколько раз продублировать последний кадр, если агент достиг цели.
    """
    print(f"Начинаем запись GIF в файл: {filename}...")

    prev_backend = matplotlib.get_backend()
    plt.switch_backend("Agg")

    frames = []
    model.eval()
    device = next(model.parameters()).device


    state, _ = env.reset()
    done = False
    terminated_flag = False

    hidden = None
    last_action = None
    last_reward = 0.0
    n_actions = env.action_space.n

    frame_count = 0

    def render_frame():
        """Рендер текущего состояния среды в numpy‑картинку и добавление в frames."""
        env.render_map()
        fig = plt.gcf()
        fig.canvas.draw()

        image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        frames.append(image)
        plt.close(fig)

    try:
        render_frame()
        frame_count += 1

        while not done and frame_count < max_frames:
            action_one_hot = np.zeros(n_actions, dtype=np.float32)
            if last_action is not None:
                action_one_hot[last_action] = 1.0

            full_input = np.concatenate(
                [state.astype(np.float32), action_one_hot, np.array([last_reward], dtype=np.float32)]
            )  # размер = obs_dim + n_actions + 1

            input_t = torch.from_numpy(full_input).float().view(1, 1, -1).to(device)

            if hidden is None:
                hidden = model.q_net.get_initial_state(batch_size=1, device=device)

            with torch.no_grad():
                q_values, hidden = model.q_net(input_t, hidden)
                action = int(q_values.argmax(dim=2).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            terminated_flag = bool(terminated)

            state = next_state
            last_action = action
            last_reward = reward

            render_frame()
            frame_count += 1

            if frame_count % 20 == 0:
                print(f"Обработано кадров: {frame_count}")

        # Если агент действительно достиг цели — держим последний кадр подольше
        if terminated_flag and len(frames) > 0 and hold_last > 1:
            last_frame = frames[-1]
            for _ in range(hold_last - 1):
                frames.append(last_frame.copy())

        if len(frames) > 0:
            imageio.mimsave(filename, frames, fps=fps, loop=0)
            print(f"Успешно сохранено: {filename} ({len(frames)} кадров)")
        else:
            print("Ошибка: не удалось записать ни одного кадра.")

    except Exception as e:
        print(f"Критическая ошибка при записи GIF: {e}")

    finally:
        plt.switch_backend(prev_backend)