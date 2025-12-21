"""
A2C + LSTM для среды GridWorld.


- Если env — VectorGridWorldEnv с n_envs:
    * каждый прогон _run_rollout_and_update() даёт батч из n_envs эпизодов;
    * градиент считается по всем этим эпизодам сразу (батчевый режим);
    * episodes считаются по ПОД-СРЕДАМ (эпизод = один запуск агента в одной копии мира).

- Сеть:
    * вход: obs_batch формы (batch, obs_dim);
    * LSTM hidden: (1, batch, hidden_dim);
    * actions: (batch,);
    * returns/advantages: (T, batch).


- Ранняя остановка и выбор лучших весов основаны на avg_episode_return:
    * avg_episode_return = средняя суммарная награда за эпизод,
      усреднённая по последним avg_window эпизодам;
    * если avg_episode_return не улучшается patience батчей подряд
      (после min_episodes_before_early_stop эпизодов) — обучение останавливается;
    * лучшие веса (best_state_dict) — те, при которых avg_episode_return был максимален.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import wandb


class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs_t: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        obs_t: (batch, obs_dim) или (obs_dim,)
        hidden: (1, batch, hidden_dim) или None
        """
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)

        x = self.fc(obs_t)           # (B, hidden_dim)
        x = x.unsqueeze(1)           # (B, 1, hidden_dim)

        lstm_out, new_hidden = self.lstm(x, hidden)  # (B, 1, hidden_dim)
        h = lstm_out.squeeze(1)      # (B, hidden_dim)

        logits = self.actor_head(h)             # (B, n_actions)
        value = self.critic_head(h).squeeze(-1) # (B,)

        return logits, value, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


@dataclass
class A2CConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    num_episodes: int = 5000

    max_steps_per_episode: int = 200   
    print_every: int = 100            
    avg_window: int = 100           

    patience: int = 20
    min_episodes_before_early_stop: int = 50


class A2CAgent:
    """
    Батчевый A2C (Actor-Critic) с LSTM.


    - При vector env:
        * один rollout даёт batch_size = n_envs эпизодов;
        * в обучении считаем returns/advantages размерности (T, n_envs);
        * считаем один шаг градиента по всему батчу.

    - Лучшая модель и ранняя остановка основаны на avg_episode_return:
        * avg_episode_return = среднее по последним avg_window эпизодам;
        * лучшие веса — при максимальном avg_episode_return;
        * ранняя остановка по отсутствию улучшения avg_episode_return
          patience батчей подряд (после min_episodes_before_early_stop эпизодов).
    """

    def __init__(
        self,
        env: gym.Env,
        config: A2CConfig,
        device: Optional[str] = None,
        use_wandb: bool = True,
    ):
        self.env = env
        self.config = config
        self.use_wandb = use_wandb

        full_obs, _ = env.reset()
        full_obs = np.asarray(full_obs, dtype=np.float32)

        if full_obs.ndim == 1:
            self.batch_size = 1
            self.obs_dim = int(full_obs.shape[0])
        elif full_obs.ndim == 2:
            # векторная среда: (n_envs, obs_dim)
            self.batch_size = int(full_obs.shape[0])
            self.obs_dim = int(full_obs.shape[1])
        else:
            raise ValueError(
                f"Неподдерживаемая форма наблюдения: {full_obs.shape}, "
                f"ожидалось (obs_dim,) или (n_envs, obs_dim)"
            )

        self.vectorized = self.batch_size > 1

        n_actions = env.action_space.n

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.net = ActorCriticLSTM(self.obs_dim, n_actions, hidden_dim=128).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)

        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []

        # Счётчик эпизодов на уровне под‑сред
        self.episode_idx: int = 0

        # Суммарное число шагов по всем под‑средам
        self.total_env_steps: int = 0

        self.best_avg_return: float = -float("inf")
        self.best_state_dict: Optional[Dict[str, Any]] = None

        self.no_improve_counter: int = 0

        self.update_idx: int = 0

        print(
            f"[A2C] vectorized={self.vectorized}, "
            f"batch_size={self.batch_size}, obs_dim={self.obs_dim}"
        )

    def _to_batch_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Приводит наблюдение к форме (batch_size, obs_dim).

        - одиночная среда: obs.shape=(obs_dim,) -> (1, obs_dim)
        - векторная: obs.shape=(n_envs, obs_dim) -> проверка на batch_size.
        """
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim == 1:
            if self.batch_size != 1:
                raise ValueError(
                    f"Тренировочный batch_size={self.batch_size}, "
                    f"но среда вернула 1D наблюдение. "
                    f"Для обучения на батче нужна векторная среда."
                )
            obs = obs.reshape(1, -1)
        elif obs.ndim == 2:
            if obs.shape[0] != self.batch_size or obs.shape[1] != self.obs_dim:
                raise ValueError(
                    f"Наблюдение формы {obs.shape} не совпадает с "
                    f"(batch_size, obs_dim)=({self.batch_size}, {self.obs_dim})"
                )
        else:
            raise ValueError(
                f"Ожидалось obs.ndim 1 или 2, получили {obs.ndim}"
            )
        return obs

    def train(self):
        cfg = self.config
        episodes_target = cfg.num_episodes
        last_print_ep = 0

        while self.episode_idx < episodes_target:
            self.update_idx += 1

            batch_returns, batch_lens, losses = self._run_rollout_and_update()

            n_batch = len(batch_returns)   # = batch_size
            remain = episodes_target - self.episode_idx
            n_use = min(n_batch, remain)

            for i in range(n_use):
                self.episode_returns.append(float(batch_returns[i]))
                self.episode_lengths.append(int(batch_lens[i]))
                self.episode_idx += 1
                self.total_env_steps += int(batch_lens[i])
                
            batch_mean_ret = float(np.mean(batch_returns))
            batch_mean_len = float(np.mean(batch_lens))

            window = cfg.avg_window
            if self.episode_returns:
                avg_ret = float(np.mean(self.episode_returns[-window:]))
                avg_len = float(np.mean(self.episode_lengths[-window:]))
            else:
                avg_ret = 0.0
                avg_len = 0.0

            improved = avg_ret > (self.best_avg_return + 1e-6)
            if improved:
                self.best_avg_return = avg_ret
                self.no_improve_counter = 0
                self.best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.net.state_dict().items()
                }
            else:
                if self.episode_idx >= cfg.min_episodes_before_early_stop:
                    self.no_improve_counter += 1

            if self.episode_idx - last_print_ep >= cfg.print_every:
                last_print_ep = self.episode_idx
                print(
                    f"[Episode {self.episode_idx}] "
                    f"BatchReturn(mean): {batch_mean_ret:.3f}, "
                    f"BatchLen(mean): {batch_mean_len:.1f}, "
                    f"AvgReturn({window}): {avg_ret:.3f}, "
                    f"AvgLen: {avg_len:.1f}, "
                    f"BestAvgReturn: {self.best_avg_return:.3f}, "
                    f"NoImprove: {self.no_improve_counter}, "
                    f"ActorLoss: {losses['actor_loss']:.4f}, "
                    f"CriticLoss: {losses['critic_loss']:.4f}"
                )

            if self.use_wandb and wandb.run is not None:
                log_data = {
                    "episode": self.episode_idx,
                    "env_step": self.total_env_steps,
                    "batch_mean_return": batch_mean_ret,
                    "batch_mean_len": batch_mean_len,
                    "avg_episode_return": avg_ret,
                    "avg_episode_len": avg_len,
                    "no_improve_counter": self.no_improve_counter,
                    "actor_loss": losses["actor_loss"],
                    "critic_loss": losses["critic_loss"],
                    "entropy_loss": losses["entropy_loss"],
                    "total_loss": losses["total_loss"],
                }
                wandb.log(log_data, step=self.total_env_steps)

            if (
                self.episode_idx >= cfg.min_episodes_before_early_stop and
                self.no_improve_counter >= cfg.patience
            ):
                print(
                    f"[A2C] Early stopping по avg_episode_return: "
                    f"{self.no_improve_counter} батчей без улучшения, "
                    f"эпизодов={self.episode_idx}, "
                    f"best_avg_return={self.best_avg_return:.3f}"
                )
                break

        if self.best_state_dict is not None:
            self.net.load_state_dict(self.best_state_dict)
            print(f"[A2C] Загружены лучшие веса по avg_episode_return={self.best_avg_return:.3f}")

    def _run_rollout_and_update(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Один батчевый rollout + один шаг градиента.

        Для VectorGridWorldEnv:
          - batch_size = n_envs;
          - каждый env даёт один эпизод (до цели или лимита шагов).

        Для одиночной среды:
          - batch_size = 1; это обычный одиночный A2C.
        """
        self.net.train()

        full_obs, _ = self.env.reset()
        obs = self._to_batch_obs(full_obs)                 # (B, obs_dim)
        B = self.batch_size

        obs_t = torch.from_numpy(obs).float().to(self.device)
        hidden = self.net.init_hidden(batch_size=B, device=self.device)

        alive = np.ones(B, dtype=bool)

        log_probs_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []
        entropies_list: List[torch.Tensor] = []
        rewards_list: List[torch.Tensor] = []
        dones_step_list: List[torch.Tensor] = []
        alive_mask_list: List[torch.Tensor] = []

        max_T = self.config.max_steps_per_episode

        for t in range(max_T):
            alive_mask = torch.from_numpy(alive.astype(np.float32)).to(self.device)  # (B,)
            alive_mask_list.append(alive_mask)

            logits, values, hidden = self.net(obs_t, hidden) 
            dist = Categorical(logits=logits)

            actions = dist.sample()              # (B,)
            log_prob = dist.log_prob(actions)    # (B,)
            entropy = dist.entropy()             # (B,)

            actions_np = actions.detach().cpu().numpy().astype(int)

            full_next_obs, reward_np, terminated_np, truncated_np, info = self.env.step(actions_np)

            reward_np = np.asarray(reward_np, dtype=np.float32)
            terminated_np = np.asarray(terminated_np, dtype=bool)
            truncated_np = np.asarray(truncated_np, dtype=bool)

            if reward_np.ndim == 0:
                reward_np = reward_np.reshape(1)
            if terminated_np.ndim == 0:
                terminated_np = terminated_np.reshape(1)
            if truncated_np.ndim == 0:
                truncated_np = truncated_np.reshape(1)

            if reward_np.shape[0] != B:
                raise ValueError(f"reward.shape={reward_np.shape}, ожидалось ({B},)")

            done_step = np.logical_or(terminated_np, truncated_np) & alive

            rewards_list.append(torch.from_numpy(reward_np).to(self.device))             # (B,)
            log_probs_list.append(log_prob)                                             # (B,)
            values_list.append(values)                                                  # (B,)
            entropies_list.append(entropy)                                              # (B,)
            dones_step_list.append(
                torch.from_numpy(done_step.astype(np.float32)).to(self.device)
            )                                                                           # (B,)

            alive[done_step] = False

            next_obs = self._to_batch_obs(full_next_obs)
            obs_t = torch.from_numpy(next_obs).float().to(self.device)

            if not alive.any():
                break

        log_probs_t = torch.stack(log_probs_list, dim=0)    # (T, B)
        values_t = torch.stack(values_list, dim=0)          # (T, B)
        entropies_t = torch.stack(entropies_list, dim=0)    # (T, B)
        rewards_t = torch.stack(rewards_list, dim=0)        # (T, B)
        dones_step_t = torch.stack(dones_step_list, dim=0)  # (T, B)
        alive_mask_t = torch.stack(alive_mask_list, dim=0)  # (T, B)

        T = log_probs_t.shape[0]

        gamma = self.config.gamma
        returns_t = torch.zeros_like(rewards_t, device=self.device)  # (T, B)
        R = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            done_here = dones_step_t[t]  
            R = rewards_t[t] + gamma * R * (1.0 - done_here)
            returns_t[t] = R

        advantages_t = returns_t - values_t  # (T, B)

        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std(unbiased=False) + 1e-8
        advantages_norm = (advantages_t - adv_mean) / adv_std

        valid_mask = alive_mask_t > 0.5       # (T, B)
        num_valid = valid_mask.sum().clamp(min=1)

        actor_loss = - (log_probs_t * advantages_norm.detach() * valid_mask).sum() / num_valid
        critic_loss = (advantages_t.pow(2) * valid_mask).sum() / num_valid
        entropy_loss = (entropies_t * valid_mask).sum() / num_valid

        total_loss = (
            actor_loss
            + self.config.value_coef * critic_loss
            - self.config.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        rewards_np = rewards_t.detach().cpu().numpy()      # (T, B)
        dones_np = dones_step_t.detach().cpu().numpy()     # (T, B)

        total_r_per_env = rewards_np.sum(axis=0)           # (B,)
        batch_returns = total_r_per_env.astype(np.float32)

        ep_lens = np.zeros(B, dtype=np.int32)
        for e in range(B):
            done_indices = np.where(dones_np[:, e] > 0.5)[0]
            if len(done_indices) > 0:
                ep_lens[e] = int(done_indices[0]) + 1
            else:
                ep_lens[e] = T  

        losses = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "total_loss": float(total_loss.item()),
        }

        return batch_returns, ep_lens, losses

    def act_greedy(
        self,
        obs: np.ndarray,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Жадное действие по текущей политике.

        obs:
          - shape (obs_dim,)  → одиночная среда;
          - shape (n_envs, obs_dim) → векторная среда с n_envs копиями.

        Возвращает:
          actions:
            - shape (,)  если был одиночный obs (скалярный int),
            - shape (n_envs,) если батч;
          new_hidden: (h, c) с batch_size, соответствующим obs.
        """
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim == 1:
            B_eval = 1
            obs_batch = obs.reshape(1, -1)
        elif obs.ndim == 2:
            B_eval, D = obs.shape
            if D != self.obs_dim:
                raise ValueError(
                    f"act_greedy: obs_dim={D} не совпадает с обученной obs_dim={self.obs_dim}"
                )
            obs_batch = obs
        else:
            raise ValueError(
                f"act_greedy: ожидалось obs.ndim 1 или 2, получили {obs.ndim}"
            )

        self.net.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_batch).float().to(self.device)
            if hidden is None:
                hidden = self.net.init_hidden(batch_size=B_eval, device=self.device)
            logits, value, new_hidden = self.net(obs_t, hidden)
            actions = torch.argmax(logits, dim=-1)   # (B_eval,)
        self.net.train()

        actions_np = actions.cpu().numpy().astype(int)
        if B_eval == 1:
            return int(actions_np[0]), new_hidden
        else:
            return actions_np, new_hidden


def evaluate_policy_success(
    env: gym.Env,
    agent_or_model: Any,
    max_steps: int,
) -> bool:
    """
    Прогоняет один эпизод жадно по текущей политике и проверяет,
    достиг ли агент цели (terminated=True хотя бы раз).

    Работает как с одиночной средой, так и с векторной:
      - если terminated — скаляр: берётся как есть;
      - если массив: успех = any(terminated).
    """
    full_obs, _ = env.reset()
    obs = np.asarray(full_obs, dtype=np.float32)

    hidden: Optional[Tuple] = None
    terminated_flag = False

    for t in range(max_steps):
        try:
            action, hidden = agent_or_model.act_greedy(obs, hidden=hidden, env=env)
        except TypeError:
            action, hidden = agent_or_model.act_greedy(obs, hidden=hidden)

        full_next_obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.asarray(full_next_obs, dtype=np.float32)

        term = np.asarray(terminated)
        trunc = np.asarray(truncated)

        if term.ndim == 0:
            term_any = bool(term)
        else:
            term_any = bool(term.any())

        if trunc.ndim == 0:
            trunc_any = bool(trunc)
        else:
            trunc_any = bool(trunc.any())

        if term_any:
            terminated_flag = True

        if term_any or trunc_any:
            break

    return terminated_flag