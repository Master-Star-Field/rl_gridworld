"""
drqn_gridworld_improved.py

Улучшенный DRQN‑агент для частично наблюдаемой среды GridWorld
с dueling‑архитектурой + Double DQN, логированием в Weights & Biases
и сохранением тестового эпизода в GIF.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.rl_grid_world.envs.grid_world import GridWorldEnv
from src.rl_grid_world.utils.save_gif import save_episode_gif
from src.rl_grid_world.utils.generate_wall import generate_walls


class DRQN(nn.Module):
    r"""# DRQN: Deep Recurrent Q‑Network (Dueling + LSTM)

    Рекуррентная Q‑сеть для частично наблюдаемых сред (POMDP).

    Архитектура:
    
    - вход: `[obs_t, one_hot(a_{t-1}), r_{t-1}]`;
    - FC‑слой → LSTM по времени;
    - dueling‑голова: `V(h_t)` и `A(h_t, a)`,  
      `Q(h_t, a) = V(h_t) + A(h_t, a) - mean_a A(h_t, a)`.

    Это сочетает:
    
    - рекуррентную память (как в DRQN / R2D2),
    - устойчивость dueling DQN,
    - возможность Double DQN на уровне Lightning‑модуля.
    """

    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )


        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.value_head = nn.Linear(hidden_dim, 1)
        self.adv_head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        r"""Вычисляет Q‑значения для последовательности входов.

        Parameters

        x : torch.Tensor
            Тензор формы `(B, T, input_dim)` — батч последовательностей.
        hidden : (h_0, c_0), optional
            Начальное скрытое состояние LSTM, формы `(1, B, hidden_dim)`.

        Returns

        q_values : torch.Tensor
            Тензор формы `(B, T, n_actions)` — Q‑значения для каждого шага.
        new_hidden : (h_T, c_T)
            Финальное скрытое состояние LSTM.
        """
        batch_size, seq_len, _ = x.size()

        x_flat = x.view(-1, x.size(2))
        x_emb = self.fc(x_flat)
        x_seq = x_emb.view(batch_size, seq_len, -1)

        lstm_out, new_hidden = self.lstm(x_seq, hidden)
        value = self.value_head(lstm_out)              # (B, T, 1)
        advantage = self.adv_head(lstm_out)           # (B, T, n_actions)
        advantage_mean = advantage.mean(dim=2, keepdim=True)
        q_values = value + (advantage - advantage_mean)

        return q_values, new_hidden

    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        r"""Создаёт нулевое начальное состояние LSTM.

        Parameters

        batch_size : int
            Размер батча.
        device : torch.device
            Устройство 

        Returns

        (h_0, c_0) : Tuple[torch.Tensor, torch.Tensor]
            Нулевые тензоры формы (1, batch_size, hidden_dim).
        """
        
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h0, c0


class EpisodeReplayBuffer:
    
    r"""# EpisodeReplayBuffer

    Буфер воспроизведения, хранящий **целые эпизоды** и возвращающий
    случайные фрагменты фиксированной длины для обучения DRQN.

    Каждый эпизод — это список переходов:

    ```python
    (state_input, action, reward, next_state_input, done_true)
    ```

    где:
    
    - `state_input` и `next_state_input` — уже сформированные вектора
      `[obs_t, one_hot(a_{t-1}), r_{t-1}]`,
    - `done_true` — 1.0 только при *настоящем* терминале (достижение цели),
      а не при обрезке по тайм‑ауту.

    При выборке:
    - если эпизод короче `seq_len`, он паддится последним переходом;
    - возвращается также маска `valid_mask`, отмечающая настоящие шаги,
      чтобы не учитывать паддинг в функции потерь.
    """

    def __init__(self, capacity: int, seq_len: int) -> None:
        
        """
        Parameters
        ----------
        capacity : int
            Максимальное количество эпизодов в буфере.
        seq_len : int
            Длина последовательности, отдаваемой при выборке.
        """
        self.buffer: List[List[Tuple[np.ndarray, int, float, np.ndarray, float]]] = []
        self.capacity = capacity
        self.seq_len = seq_len

    def push_episode(self, episode: List[Tuple[np.ndarray, int, float, np.ndarray, float]]) -> None:
        """Добавляет один завершённый эпизод в буфер (FIFO по эпизодам)."""
        if len(episode) == 0:
            return
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def __len__(self) -> int:
        """Возвращает количество эпизодов в буфере."""
        return len(self.buffer)

    def sample_sequences(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Сэмплирует батч последовательностей длины `seq_len`.

        Parameters

        batch_size : int
            Размер батча (количество эпизодов/последовательностей).

        Returns

        states : np.ndarray
            '(B, T, input_dim)' — входы в Q‑сетку.
        actions : np.ndarray
            '(B, T)' — индексы действий.
        rewards : np.ndarray
            '(B, T)' — награды.
        next_states : np.ndarray
            '(B, T, input_dim)' — входы для следующего шага.
        dones : np.ndarray
            '(B, T)' — флаг терминала (1.0 только для истинного терминала).
        mask : np.ndarray
            '(B, T)' — 1.0 для реальных шагов, 0.0 для паддинга.
        """
        batch_s, batch_a, batch_r, batch_ns, batch_d, batch_mask = [], [], [], [], [], []

        for _ in range(batch_size):
            episode = random.choice(self.buffer)
            ep_len = len(episode)

            if ep_len >= self.seq_len:
                start = np.random.randint(0, ep_len - self.seq_len + 1)
                seq_slice = episode[start : start + self.seq_len]
                valid_len = self.seq_len
            else:
                seq_slice = episode[:]
                valid_len = ep_len
                padding = [episode[-1]] * (self.seq_len - ep_len)
                seq_slice.extend(padding)

            s, a, r, ns, d = zip(*seq_slice)

            batch_s.append(np.array(s, dtype=np.float32))
            batch_a.append(np.array(a, dtype=np.int64))
            batch_r.append(np.array(r, dtype=np.float32))
            batch_ns.append(np.array(ns, dtype=np.float32))
            batch_d.append(np.array(d, dtype=np.float32))

            m = np.zeros(self.seq_len, dtype=np.float32)
            m[:valid_len] = 1.0
            batch_mask.append(m)

        return (
            np.stack(batch_s, axis=0),
            np.stack(batch_a, axis=0),
            np.stack(batch_r, axis=0),
            np.stack(batch_ns, axis=0),
            np.stack(batch_d, axis=0),
            np.stack(batch_mask, axis=0),
        )



class DummyDataset(IterableDataset):
    r"""# DummyDataset

    Бесконечный датасет для PyTorch Lightning, используемый в задачах RL.

    Lightning ожидает `DataLoader`, но в on‑policy/off‑policy RL
    данные генерируются самим агентом в среде, поэтому этот датасет
    просто генерирует фиктивные элементы, чтобы триггерить `training_step`
    нужное количество раз.
    """

    def __iter__(self):
        while True:
            yield torch.tensor(0, dtype=torch.int64)



class DRQNLightning(pl.LightningModule):
    r"""# DRQNLightning

    Lightning‑обёртка над DRQN‑агентом для обучения в среде `GridWorldEnv`
    с dueling‑архитектурой и Double DQN, используя ручную оптимизацию.

    Цикл:

    1. На каждом `training_step` агент делает один шаг в среде.
    2. Переход добавляется в текущий эпизод.
    3. При завершении эпизода — эпизод сохраняется в replay‑буфер.
    4. Когда в буфере достаточно эпизодов, из него сэмплируются последовательности
       и выполняется один градиентный шаг DRQN.
    """

    def __init__(
        self,
        env_params: Dict[str, Any],
        lr: float = 1e-3,
        gamma: float = 0.99,
        seq_len: int = 20,
        burn_in: int = 5,
        batch_size: int = 32,
        buffer_size: int = 500,
        min_episodes: int = 10,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        sync_rate: int = 1_000,
        hidden_dim: int = 128,
        avg_window: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        #ручная оптимизация
        self.automatic_optimization = False

        self.env = GridWorldEnv(**env_params)
        obs, _ = self.env.reset()
        self.obs_dim = int(obs.shape[0])
        self.n_actions = int(self.env.action_space.n)

        # obs + one-hot(prev_action) + prev_reward
        self.input_dim = self.obs_dim + self.n_actions + 1

        # Q‑сеть
        self.q_net = DRQN(self.input_dim, self.n_actions, hidden_dim=hidden_dim)
        self.target_net = DRQN(self.input_dim, self.n_actions, hidden_dim=hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.buffer = EpisodeReplayBuffer(capacity=buffer_size, seq_len=seq_len)
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.batch_size = batch_size

        self.state = obs
        self.last_action: Optional[int] = None
        self.last_reward: float = 0.0
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.episode_reward: float = 0.0
        self.episode_len: int = 0
        self.history: List[Tuple[np.ndarray, int, float, np.ndarray, float]] = []

        self.total_env_steps: int = 0
        self.total_episodes: int = 0
        self.learn_steps: int = 0

        self.recent_returns: deque[float] = deque(maxlen=avg_window)
        self.recent_lengths: deque[int] = deque(maxlen=avg_window)

        self.epsilon = epsilon_start

    def create_input(
        self,
        obs: np.ndarray,
        last_action_idx: Optional[int],
        last_reward: float,
    ) -> np.ndarray:
        
        r"""Формирует входной вектор `[obs, one_hot(a_{t-1}), r_{t-1}]`.

        Пустое предыдущее действие (`None`) кодируется нулевым one‑hot вектором.
        """
        action_vec = np.zeros(self.n_actions, dtype=np.float32)
        if last_action_idx is not None:
            action_vec[last_action_idx] = 1.0

        reward_vec = np.array([last_reward], dtype=np.float32)
        return np.concatenate([obs.astype(np.float32), action_vec, reward_vec], axis=0)

    def get_action(
        self,
        obs: np.ndarray,
        last_action_idx: Optional[int],
        last_reward: float,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
        epsilon: float,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        
        r"""Выбирает действие по ε‑greedy политике и обновляет скрытое состояние LSTM."""
        input_vec = self.create_input(obs, last_action_idx, last_reward)
        state_t = torch.from_numpy(input_vec).float().view(1, 1, -1).to(self.device)

        if hidden is None:
            hidden = self.q_net.get_initial_state(batch_size=1, device=self.device)

        with torch.no_grad():
            q_vals, new_hidden = self.q_net(state_t, hidden)

        if np.random.rand() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(q_vals.argmax(dim=-1).item())

        h, c = new_hidden
        new_hidden = (h.detach(), c.detach())
        return action, new_hidden


    def configure_optimizers(self):
        """Создаёт Adam‑оптимизатор для параметров основной Q‑сети."""
        optimizer = optim.Adam(self.q_net.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        """Возвращает бесконечный DataLoader для вызова `training_step`."""
        return DataLoader(DummyDataset(), batch_size=1, num_workers=0)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        r"""Один шаг обучения:

        1. Делаем шаг в среде (acting).
        2. При завершении эпизода — кладём эпизод в буфер.
        3. Если буфер достаточно заполнен — выполняем градиентный шаг DRQN (Double DQN).
        """
        opt = self.optimizers()

        action, self.hidden_state = self.get_action(
            self.state,
            self.last_action,
            self.last_reward,
            self.hidden_state,
            self.epsilon,
        )

        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done_env = terminated or truncated
        done_true = float(terminated)

        current_input = self.create_input(self.state, self.last_action, self.last_reward)
        next_input = self.create_input(next_obs, action, reward)

        self.history.append((current_input, action, reward, next_input, done_true))

        self.state = next_obs
        self.last_action = action
        self.last_reward = reward
        self.episode_reward += reward
        self.episode_len += 1
        self.total_env_steps += 1

        eps_start = self.hparams.epsilon_start
        eps_end = self.hparams.epsilon_end
        eps_decay_steps = max(1, self.hparams.epsilon_decay)

        frac = min(1.0, self.total_env_steps / eps_decay_steps)
        self.epsilon = eps_start + frac * (eps_end - eps_start)

        self.log("epsilon", self.epsilon, prog_bar=True, on_step=True, logger=True)
        self.log("env_step", self.total_env_steps, prog_bar=False, on_step=True, logger=True)

        if done_env:
            self.buffer.push_episode(self.history)
            self.total_episodes += 1

            self.recent_returns.append(self.episode_reward)
            self.recent_lengths.append(self.episode_len)

            avg_return = float(np.mean(self.recent_returns))
            avg_len = float(np.mean(self.recent_lengths))

            self.log("episode_return", self.episode_reward, prog_bar=True, on_step=True, logger=True)
            self.log("episode_len", self.episode_len, prog_bar=False, on_step=True, logger=True)
            self.log("episode_success", int(done_true > 0.0), prog_bar=True, on_step=True, logger=True)
            self.log("avg_episode_return", avg_return, prog_bar=True, on_step=True, logger=True)
            self.log("avg_episode_len", avg_len, prog_bar=False, on_step=True, logger=True)

            obs, _ = self.env.reset()
            self.state = obs
            self.last_action = None
            self.last_reward = 0.0
            self.hidden_state = None
            self.episode_reward = 0.0
            self.episode_len = 0
            self.history = []

        if len(self.buffer) >= self.hparams.min_episodes:
            (
                s_np,
                a_np,
                r_np,
                ns_np,
                d_np,
                mask_np,
            ) = self.buffer.sample_sequences(self.batch_size)

            states = torch.from_numpy(s_np).float().to(self.device)       # (B, T, input_dim)
            actions = torch.from_numpy(a_np).long().to(self.device)       # (B, T)
            rewards = torch.from_numpy(r_np).float().to(self.device)      # (B, T)
            next_states = torch.from_numpy(ns_np).float().to(self.device) # (B, T, input_dim)
            dones = torch.from_numpy(d_np).float().to(self.device)        # (B, T)
            mask = torch.from_numpy(mask_np).float().to(self.device)      # (B, T)

            B, T, _ = states.shape

            # burn-in: первые шаги только для прогрева LSTM
            burn_in = min(self.burn_in, T)
            if burn_in > 0:
                mask[:, :burn_in] = 0.0

            hidden_init = self.q_net.get_initial_state(B, self.device)
            q_seq, _ = self.q_net(states, hidden_init)
            q_taken = q_seq.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (B, T)

            with torch.no_grad():
                online_hidden = self.q_net.get_initial_state(B, self.device)
                q_next_online, _ = self.q_net(next_states, online_hidden)
                next_actions = q_next_online.argmax(dim=2)  # (B, T)

                target_hidden = self.target_net.get_initial_state(B, self.device)
                q_next_target, _ = self.target_net(next_states, target_hidden)
                q_next_double = q_next_target.gather(
                    2, next_actions.unsqueeze(-1)
                ).squeeze(-1)  # (B, T)

                targets = rewards + self.hparams.gamma * q_next_double * (1.0 - dones)

            td_error = q_taken - targets
            loss = (td_error.pow(2) * mask).sum() / (mask.sum() + 1e-8)

            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

            self.learn_steps += 1

            self.log("train_loss", loss, prog_bar=True, on_step=True, logger=True)
            self.log("learn_steps", self.learn_steps, prog_bar=False, on_step=True, logger=True)

            if self.learn_steps % self.hparams.sync_rate == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        return None


def train_drqn_gridworld(
    env_params: Dict[str, Any],
    max_steps: int = 50_000,
    project: str = "gridworld_drqn",
    run_name: Optional[str] = None,
    gif_path: str = "drqn_eval_episode.gif",
    **agent_kwargs: Any,
) -> None:
    r"""# train_drqn_gridworld

    Высокоуровневая функция для запуска обучения DRQN‑агента
    в среде `GridWorldEnv` с логированием в Weights & Biases
    и сохранением одного тестового эпизода в GIF.

    Parameters
    ----------
    env_params : dict
        Параметры инициализации среды `GridWorldEnv`.
    max_steps : int
        Максимальное количество шагов обучения (и шагов среды).
    project : str
        Имя проекта в Weights & Biases.
    run_name : str, optional
        Имя конкретного запуска (run) в Weights & Biases.
    gif_path : str
        Путь для сохранения GIF с тестовым эпизодом.
    **agent_kwargs : Any
        Дополнительные параметры для `DRQNLightning` (lr, gamma, seq_len, ...).
    """
    model = DRQNLightning(env_params=env_params, **agent_kwargs)

    wandb_logger = WandbLogger(
        project=project,
        name=run_name,
        log_model=False,
    )

    trainer = pl.Trainer(
        max_steps=max_steps,
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model)

    eval_env_params = dict(env_params)
    eval_env_params["render_mode"] = "rgb_array"
    eval_env = GridWorldEnv(**eval_env_params)

    save_episode_gif(eval_env, model, gif_path)
