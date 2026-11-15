
"""DQN with optional shield-aware targets and Double DQN."""

from collections import deque
import random

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        seed=42,
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        capacity=10000,
        backup_mode="standard",
    ):
        if backup_mode not in ("standard", "shield_safe"):
            raise ValueError(
                "backup_mode must be 'standard' or 'shield_safe'"
            )

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.online = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=lr
        )

        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.updates = 0
        self.backup_mode = backup_mode
        self.last_stats = {}

    def q_values(self, observation):
        with torch.no_grad():
            x = torch.as_tensor(
                observation, dtype=torch.float32
            ).unsqueeze(0)

            return self.online(x)[0].cpu().numpy()

    def act(self, observation, epsilon=0.0):
        if self.rng.random() < epsilon:
            return self.rng.randrange(4)

        return int(np.argmax(self.q_values(observation)))

    def remember(
        self,
        state,
        executed_action,
        reward,
        next_state,
        done,
        next_safe_actions=None,
    ):
        if self.backup_mode == "shield_safe" and not done:
            if next_safe_actions is None or len(next_safe_actions) == 0:
                raise ValueError(
                    "shield_safe requires next_safe_actions"
                )

        mask = np.zeros(4, dtype=np.bool_)

        if next_safe_actions is None:
            mask[:] = True
        else:
            mask[list(next_safe_actions)] = True

        self.memory.append((
            np.array(state, copy=True),
            int(executed_action),
            float(reward),
            np.array(next_state, copy=True),
            bool(done),
            mask,
        ))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.rng.sample(
            self.memory, self.batch_size
        )

        states = torch.as_tensor(
            np.stack([t[0] for t in batch]),
            dtype=torch.float32,
        )

        actions = torch.as_tensor(
            [t[1] for t in batch],
            dtype=torch.int64,
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            [t[2] for t in batch],
            dtype=torch.float32,
        )

        next_states = torch.as_tensor(
            np.stack([t[3] for t in batch]),
            dtype=torch.float32,
        )

        dones = torch.as_tensor(
            [t[4] for t in batch],
            dtype=torch.bool,
        )

        masks = torch.as_tensor(
            np.stack([t[5] for t in batch]),
            dtype=torch.bool,
        )

        current = self.online(states).gather(
            1, actions
        ).squeeze(1)

        with torch.no_grad():

            # Double DQN:
            # Online network selects the next action.
            # Target network estimates its value.

            next_online = self.online(next_states)

            if self.backup_mode == "shield_safe":
                # Unsafe actions are excluded from selection.
                # Terminal transitions do not bootstrap.
                eligible = masks | dones.unsqueeze(1)

                next_online = next_online.masked_fill(
                    ~eligible, -torch.inf
                )

            next_actions = next_online.argmax(
                dim=1, keepdim=True
            )

            next_values = self.target(
                next_states
            ).gather(
                1, next_actions
            ).squeeze(1)

            targets = (
                rewards
                + self.gamma
                * next_values
                * (~dones).float()
            )

        loss = nn.functional.smooth_l1_loss(
            current, targets
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Nonfinite DQN loss: aborting experiment"
            )

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online.parameters(), 10.0
        )

        self.optimizer.step()
        self.updates += 1

        if self.updates % 100 == 0:
            self.target.load_state_dict(
                self.online.state_dict()
            )

        self.last_stats = {
            "q_max": float(
                current.detach().max().item()
            ),
            "target_max": float(
                targets.max().item()
            ),
            "loss": float(loss.item()),
        }

        return float(loss.item())
