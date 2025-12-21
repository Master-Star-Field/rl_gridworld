import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import imageio
from typing import Optional


def save_episode_gif(
    env,
    model,
    filename: str = "episode.gif",
    fps: int = 2,
    max_frames: int = 200,
    hold_last: int = 10,
) -> bool:
    """
    Сохраняет GIF одного эпизода

    Возвращает:
      True  — если в эпизоде хотя бы раз был terminated=True,
      False — иначе (сохранён будет последний эпизод).
    """
    print(f"Начинаем запись GIF в файл: {filename}...")

    prev_backend = matplotlib.get_backend()
    plt.switch_backend("Agg")

    frames = []
    use_agent = False
    agent = None
    net = None

    if hasattr(model, "act_greedy"):
        agent = model
        device = getattr(model, "device", torch.device("cpu"))
        use_agent = True
    elif hasattr(model, "net"):
        agent = model
        net = agent.net
        device = agent.device
    else:
        agent = None
        net = model
        device = next(net.parameters()).device

    if net is not None:
        net.eval()

    def render_frame():
        # Векторная среда
        if hasattr(env, "_render_map"):
            img = env._render_map(return_rgb_array=True)
            if img is not None:
                frames.append(img)
            return

        # Обычная GridWorldEnv
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

    try:
        full_obs, _ = env.reset()
        obs = np.asarray(full_obs, dtype=np.float32)

        if obs.ndim == 1:
            n_envs = 1
        elif obs.ndim == 2:
            n_envs = obs.shape[0]
        else:
            raise ValueError(
                f"save_episode_gif: ожидалось obs.ndim 1 или 2, получили {obs.ndim}"
            )

        done_vec = np.zeros(n_envs, dtype=bool)
        terminated_flag = False

        hidden = None
        frame_count = 0

        render_frame()
        frame_count += 1

        while (not done_vec.all()) and frame_count < max_frames:
            if use_agent:
                try:
                    action, hidden = agent.act_greedy(obs, hidden=hidden, env=env)
                except TypeError:
                    action, hidden = agent.act_greedy(obs, hidden=hidden)
            else:
                obs_np = np.asarray(obs, dtype=np.float32)
                if obs_np.ndim == 1:
                    B = 1
                    obs_batch = obs_np.reshape(1, -1)
                elif obs_np.ndim == 2:
                    B = obs_np.shape[0]
                    obs_batch = obs_np
                else:
                    raise ValueError(
                        f"save_episode_gif: ожидалось obs.ndim 1 или 2, получили {obs_np.ndim}"
                    )

                if hasattr(net, "init_hidden"):
                    if hidden is None or hidden[0].shape[1] != B:
                        hidden = net.init_hidden(batch_size=B, device=device)

                obs_t = torch.from_numpy(obs_batch).float().to(device)

                with torch.no_grad():
                    if hidden is not None:
                        out = net(obs_t, hidden)
                    else:
                        out = net(obs_t)

                    if isinstance(out, tuple) and len(out) == 3:
                        logits, value, hidden = out
                    elif isinstance(out, tuple) and len(out) == 2:
                        logits, hidden = out
                        value = None
                    else:
                        logits = out
                        value = None
                        hidden = None

                    actions_tensor = torch.argmax(logits, dim=-1)  # (B,)

                actions_np = actions_tensor.cpu().numpy().astype(int)
                if B == 1:
                    action = int(actions_np[0])
                else:
                    action = actions_np

            full_next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.asarray(full_next_obs, dtype=np.float32)

            term = np.asarray(terminated)
            trunc = np.asarray(truncated)

            if term.ndim == 0:
                term = term.reshape(1)
            if trunc.ndim == 0:
                trunc = trunc.reshape(1)

            step_done = np.logical_or(term.astype(bool), trunc.astype(bool))

            if step_done.shape[0] != done_vec.shape[0]:
                if step_done.shape[0] == 1 and done_vec.shape[0] == 1:
                    pass
                else:
                    raise ValueError(
                        f"Ожидалась длина done_vec={done_vec.shape[0]}, "
                        f"но пришёл step_done.shape={step_done.shape}"
                    )

            done_vec = np.logical_or(done_vec, step_done)
            terminated_flag = terminated_flag or bool(term.any())

            render_frame()
            frame_count += 1

            if frame_count % 20 == 0:
                print(f"Обработано кадров: {frame_count}")

        if terminated_flag and len(frames) > 0 and hold_last > 1:
            last_frame = frames[-1]
            for _ in range(hold_last - 1):
                frames.append(last_frame.copy())

        if len(frames) > 0:
            imageio.mimsave(filename, frames, fps=fps, loop=1)
            print(f"Успешно сохранено: {filename} ({len(frames)} кадров)")
            return bool(terminated_flag)
        else:
            print("Ошибка: не удалось записать ни одного кадра.")
            raise RuntimeError("save_episode_gif: список frames пуст.")

    except Exception as e:
        print(f"Критическая ошибка при записи GIF: {e}")
        raise

    finally:
        plt.switch_backend(prev_backend)
        if net is not None:
            net.train()