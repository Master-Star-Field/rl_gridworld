from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl

from src.rl_grid_world.utils.save_gif import save_episode_gif


class DRQN(nn.Module):
    r"""DRQN: Deep Recurrent Q‑Network (Dueling + LSTM)"""

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
        """
        x: (B, T, input_dim)
        hidden: (1, B, hidden_dim)
        -> q_values: (B, T, n_actions), new_hidden
        """
        batch_size, seq_len, _ = x.size()

        x_flat = x.view(-1, x.size(2))
        x_emb = self.fc(x_flat)
        x_seq = x_emb.view(batch_size, seq_len, -1)

        lstm_out, new_hidden = self.lstm(x_seq, hidden)
        value = self.value_head(lstm_out)           # (B, T, 1)
        advantage = self.adv_head(lstm_out)         # (B, T, n_actions)
        advantage_mean = advantage.mean(dim=2, keepdim=True)
        q_values = value + (advantage - advantage_mean)

        return q_values, new_hidden

    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h0, c0


class EpisodeReplayBuffer:
    """Хранит целые эпизоды и выдаёт последовательности фиксированной длины."""

    def __init__(self, capacity: int, seq_len: int) -> None:
        self.buffer: List[List[Tuple[np.ndarray, int, float, np.ndarray, float]]] = []
        self.capacity = capacity
        self.seq_len = seq_len

    def push_episode(self, episode: List[Tuple[np.ndarray, int, float, np.ndarray, float]]) -> None:
        if not episode:
            return
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample_sequences(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


class DummyDataset(torch.utils.data.Dataset):
    """Датасет фиксированной длины для корректного отображения прогресса в Lightning."""

    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(0, dtype=torch.int64)


class DRQNLightning(pl.LightningModule):
    r"""DRQNLightning

    Lightning‑обёртка над DRQN‑агентом.

    Особенности:
      - Корректно работает и с одиночной GridWorldEnv, и с векторной
        (SyncVectorEnv / GymVectorGridWorldEnv, VectorGridWorldEnv), но
        обучается логически только по ОДНОЙ под‑среде (первой),
        чтобы не ломать архитектуру эпизодного replay‑буфера.
      - Вход сети упрощён: только наблюдение obs (без a_{t-1}, r_{t-1}).
      - Лучшая модель и ранняя остановка основаны на avg_episode_return:
            avg_episode_return = среднее по последним avg_window эпизодам.
        Если avg_episode_return не улучшается early_stop_patience эпизодов подряд
        (после early_stop_min_episodes) — тренировка останавливается.
      - best_state_dict хранит лучшие веса по avg_episode_return, и после
        окончания обучения мы ими инициализируем сеть.
    """

    def __init__(
        self,
        env: gym.Env,
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
        optimizer: str = "rmsprop",
        rmsprop_alpha: float = 0.95,
        rmsprop_eps: float = 0.01,
        adadelta_rho: float = 0.9,
        weight_decay: float = 0.0,
        max_grad_norm: float = 10.0,
        early_stop_threshold: float = 0.95,
        early_stop_min_episodes: int = 200,
        early_stop_patience: int = 20,
        max_steps: int = 20000

    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["env"])

        self.automatic_optimization = False
        self.env = env
        self.max_steps = max_steps
        obs, _ = self.env.reset()
        obs_arr = np.asarray(obs, dtype=np.float32)

        if obs_arr.ndim == 1:
            self.vectorized = False
            self.n_envs = 1
            first_obs = obs_arr
        elif obs_arr.ndim == 2:
            self.vectorized = True
            self.n_envs = int(obs_arr.shape[0])
            first_obs = obs_arr[0]
        else:
            raise ValueError(
                f"Ожидались наблюдения 1D или 2D, получили shape={obs_arr.shape}"
            )

        obs_vec = self._obs_to_vec(first_obs)      # (obs_dim,)
        self.obs_dim = int(obs_vec.shape[0])

        self.n_actions = int(self.env.action_space.n)
        self.input_dim = self.obs_dim

        self.q_net = DRQN(self.input_dim, self.n_actions, hidden_dim=hidden_dim)
        self.target_net = DRQN(self.input_dim, self.n_actions, hidden_dim=hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.buffer = EpisodeReplayBuffer(capacity=buffer_size, seq_len=seq_len)
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.batch_size = batch_size

        self.state = obs_vec
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

        self.best_avg_return: float = -float("inf")
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.no_improve_counter: int = 0


    @staticmethod
    def _obs_to_vec(obs: Any) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        return arr.reshape(-1)

    def create_input(self, obs: np.ndarray) -> np.ndarray:
        """
        Формирует вход для DRQN: здесь это просто вектор наблюдения obs.
        """
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def get_action(
        self,
        obs: np.ndarray,
        last_action_idx: Optional[int],
        last_reward: float,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
        epsilon: float,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Выбор действия по epsilon-greedy стратегии.
        Использует только текущее наблюдение obs как вход.
        """
        input_vec = self.create_input(obs)
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
        opt_name = self.hparams.optimizer.lower()

        if opt_name == "adam":
            optimizer = optim.Adam(
                self.q_net.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif opt_name == "adadelta":
            optimizer = optim.Adadelta(
                self.q_net.parameters(),
                lr=self.hparams.lr,
                rho=self.hparams.adadelta_rho,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = optim.RMSprop(
                self.q_net.parameters(),
                lr=self.hparams.lr,
                alpha=self.hparams.rmsprop_alpha,
                eps=self.hparams.rmsprop_eps,
                weight_decay=self.hparams.weight_decay,
            )
        return optimizer

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            DummyDataset(self.max_steps), 
            batch_size=1, 
            num_workers=0
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        opt = self.optimizers()

        action, self.hidden_state = self.get_action(
            self.state,
            last_action_idx=None,
            last_reward=0.0,
            hidden=self.hidden_state,
            epsilon=self.epsilon,
        )

        next_obs, reward, terminated, truncated, info = self.env.step(action)

        if self.vectorized:
            next_obs_arr = np.asarray(next_obs, dtype=np.float32)
            next_obs0 = next_obs_arr[0]
            reward_arr = np.asarray(reward, dtype=np.float32)
            term_arr = np.asarray(terminated)
            trunc_arr = np.asarray(truncated)

            reward_scalar = float(reward_arr[0])
            term_any = bool(term_arr[0])
            trunc_any = bool(trunc_arr[0])
            next_obs_vec = self._obs_to_vec(next_obs0)
        else:
            reward_scalar = float(reward)
            term_any = bool(terminated)
            trunc_any = bool(truncated)
            next_obs_vec = self._obs_to_vec(next_obs)

        done_env = term_any or trunc_any
        done_true = 1.0 if done_env else 0.0

        current_input = self.create_input(self.state)
        next_input = self.create_input(next_obs_vec)

        self.history.append((current_input, action, reward_scalar, next_input, done_true))

        self.state = next_obs_vec
        self.episode_reward += reward_scalar
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


            improved = avg_return > (self.best_avg_return + 1e-6)
            if improved:
                self.best_avg_return = avg_return
                self.no_improve_counter = 0
                self.best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.q_net.state_dict().items()
                }
            else:
                if self.total_episodes >= self.hparams.early_stop_min_episodes:
                    self.no_improve_counter += 1

            self.log(
                "best_avg_episode_return",
                self.best_avg_return,
                prog_bar=True,
                on_step=True,
                logger=True,
            )
            self.log(
                "no_improve_counter",
                self.no_improve_counter,
                prog_bar=False,
                on_step=True,
                logger=True,
            )

            if (
                self.total_episodes >= self.hparams.early_stop_min_episodes
                and self.no_improve_counter >= self.hparams.early_stop_patience
                and self.trainer is not None
                and not getattr(self.trainer, "_early_stop_triggered", False)
            ):
                print(
                    f"[DRQN] Early stopping по avg_episode_return: "
                    f"{self.no_improve_counter} эпизодов без улучшения, "
                    f"total_episodes={self.total_episodes}, "
                    f"best_avg_return={self.best_avg_return:.3f}"
                )
                self.trainer.should_stop = True
                setattr(self.trainer, "_early_stop_triggered", True)
                self.log("early_stop_triggered", 1.0, prog_bar=True, on_step=True, logger=True)

            obs, _ = self.env.reset()
            obs_arr = np.asarray(obs, dtype=np.float32)
            if self.vectorized:
                first_obs = obs_arr[0]
            else:
                first_obs = obs_arr
            self.state = self._obs_to_vec(first_obs)
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

            states = torch.from_numpy(s_np).float().to(self.device)        # (B, T, input_dim)
            actions = torch.from_numpy(a_np).long().to(self.device)        # (B, T)
            rewards = torch.from_numpy(r_np).float().to(self.device)       # (B, T)
            next_states = torch.from_numpy(ns_np).float().to(self.device)  # (B, T, input_dim)
            dones = torch.from_numpy(d_np).float().to(self.device)         # (B, T)
            mask = torch.from_numpy(mask_np).float().to(self.device)       # (B, T)

            B, T, _ = states.shape

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

            if self.hparams.max_grad_norm is not None and self.hparams.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.hparams.max_grad_norm)

            opt.step()

            self.learn_steps += 1

            self.log("train_loss", loss, prog_bar=True, on_step=True, logger=True)
            self.log("learn_steps", self.learn_steps, prog_bar=False, on_step=True, logger=True)

            if self.learn_steps % self.hparams.sync_rate == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        return None


    def act_greedy(
        self,
        obs: np.ndarray,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        env: Optional[gym.Env] = None,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Жадное действие по Q‑сетке (ε=0).

        obs может быть:
          - (obs_dim,)
          - (n_envs, obs_dim) — берём первую под‑среду.
        """
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 2:
            obs_vec = self._obs_to_vec(obs_arr[0])
        else:
            obs_vec = self._obs_to_vec(obs_arr)

        action, new_hidden = self.get_action(
            obs_vec,
            last_action_idx=None,
            last_reward=0.0,
            hidden=hidden,
            epsilon=0.0,
        )
        return action, new_hidden