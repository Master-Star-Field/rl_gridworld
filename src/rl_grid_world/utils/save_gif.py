import imageio
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

def save_episode_gif(model, env, filename="episode.gif", fps=5):
    """
    Сохраняет GIF эпизода без открытия окна игры.
    """
    print(f"Начинаем запись GIF в файл: {filename}...")
    

    prev_backend = matplotlib.get_backend()
    plt.switch_backend('Agg')
    
    frames = []
    
    model.eval()
    device = model.device
    state, _ = env.reset()
    done = False
    hidden = None
    last_action = 0
    n_actions = env.action_space.n
    
    max_frames = 200 
    frame_count = 0
    
    try:
        while not done and frame_count < max_frames:

            env.render_map()
            fig = plt.gcf()
            

            fig.canvas.draw()
            image_flat = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            
            image = image_flat[:, :, :3]
            frames.append(image)
            plt.close(fig)
            

            action_one_hot = np.zeros(n_actions, dtype=np.float32)
            action_one_hot[last_action] = 1.0
            full_input = np.concatenate([state, action_one_hot])
            
            input_t = torch.FloatTensor(full_input).view(1, 1, -1).to(device)
            
            with torch.no_grad():
                q_values, hidden = model.q_net(input_t, hidden)
                action = q_values.argmax(dim=2).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            last_action = action
            frame_count += 1
            
            if frame_count % 20 == 0:
                print(f"Обработано кадров: {frame_count}")

        if len(frames) > 0:
            imageio.mimsave(filename, frames, fps=fps, loop=0)
            print(f"Успешно сохранено: {filename} ({len(frames)} кадров)")
        else:
            print("Ошибка: не удалось записать ни одного кадра.")
            
    except Exception as e:
        print(f"Критическая ошибка при записи GIF: {e}")
        
    finally:

        plt.switch_backend(prev_backend)
