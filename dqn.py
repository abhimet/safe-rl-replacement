
"""DQN with optional shield-aware targets and Double DQN."""

from collections import deque
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)


class SpatialQNetwork(nn.Module):
    """Q-network with location-aware features derived from the supplied map.

    This is not a shield: no actions are masked, replaced, or modified.
    The egocentric patch and one-hot position/goal are deterministic functions
    of the agent's existing (position, goal, occupancy grid) observation.
    """

    def __init__(self, input_dim=104):
        super().__init__()
        cells = input_dim - 4
        size = int(cells ** .5)
        if size < 2 or size * size != cells:
            raise ValueError("spatial architecture needs map observation (4 + square grid)")
        self.size = size
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2 * cells + 27, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, observation):
        one = observation.ndim == 1
        x = observation.unsqueeze(0) if one else observation
        b, n = x.shape
        if n != self.input_dim:
            raise ValueError(f"expected {self.input_dim} observation features, got {n}")
        size = self.size
        row = (x[:, 0] * (size - 1)).round().long().clamp(0, size - 1)
        col = (x[:, 1] * (size - 1)).round().long().clamp(0, size - 1)
        goal_row = (x[:, 2] * (size - 1)).round().long().clamp(0, size - 1)
        goal_col = (x[:, 3] * (size - 1)).round().long().clamp(0, size - 1)
        agent_plane = F.one_hot(row * size + col, size * size).to(x.dtype)
        goal_plane = F.one_hot(goal_row * size + goal_col, size * size).to(x.dtype)
        walls = F.pad(x[:, 4:].reshape(b, 1, size, size), (2, 2, 2, 2), value=1.0)
        indices = torch.arange(5, device=x.device)
        patch_indices = (
            (row[:, None, None] + indices[None, :, None]) * (size + 4)
            + (col[:, None, None] + indices[None, None, :])
        ).reshape(b, 25)
        local_walls = walls.reshape(b, -1).gather(1, patch_indices)
        relative_goal = torch.stack((goal_row - row, goal_col - col), dim=1).to(x.dtype) / (size - 1)
        z = torch.cat((x, agent_plane, goal_plane, local_walls, relative_goal), dim=1)
        values = self.net(z)
        return values[0] if one else values


def create_network(input_dim, architecture="mlp"):
    if architecture == "mlp":
        return QNetwork(input_dim=input_dim)
    if architecture == "spatial":
        return SpatialQNetwork(input_dim=input_dim)
    raise ValueError(f"Unknown Q-network architecture: {architecture}")


def network_from_weights(weights):
    """Load both historical raw state_dict and new spatial-model checkpoints."""
    dim = int(weights["net.0.weight"].shape[1])
    # Spatial first-layer dimensionality is 31 + 3 * grid_area.
    area = (dim - 31) // 3
    side = int(area ** .5) if area > 0 else 0
    if dim >= 43 and (dim - 31) % 3 == 0 and side * side == area:
        model = create_network(4 + area, "spatial")
    else:
        model = create_network(dim, "mlp")
    model.load_state_dict(weights)
    model.eval()
    return model


class DQNAgent:
    def __init__(
        self,
        seed=42,
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        capacity=10000,
        backup_mode="standard",
        input_dim=2,
        safety_loss_weight=0.0,
        safety_margin=1.0,
        architecture="mlp",
        expert_loss_weight=0.0,
    ):
        if backup_mode not in ("standard", "shield_safe"):
            raise ValueError(
                "backup_mode must be 'standard' or 'shield_safe'"
            )

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.online = create_network(input_dim, architecture)
        self.target = create_network(input_dim, architecture)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=lr
        )

        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.updates = 0
        self.backup_mode = backup_mode
        if not np.isfinite(safety_loss_weight) or safety_loss_weight < 0:
            raise ValueError("safety_loss_weight must be finite and nonnegative")
        if not np.isfinite(safety_margin) or safety_margin <= 0:
            raise ValueError("safety_margin must be finite and positive")
        self.safety_loss_weight = float(safety_loss_weight)
        self.safety_margin = float(safety_margin)
        if not np.isfinite(expert_loss_weight) or expert_loss_weight < 0:
            raise ValueError("expert_loss_weight must be finite and nonnegative")
        self.expert_loss_weight = float(expert_loss_weight)
        self.expert_dataset = None
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
        current_safe_actions=None,
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

        # The shield provides labels at training time only. No mask is used
        # to select actions during unshielded evaluation.
        if self.safety_loss_weight > 0 and current_safe_actions is None:
            raise ValueError("Safety feedback requires current_safe_actions")
        current_mask = np.zeros(4, dtype=np.bool_)
        if current_safe_actions is None:
            current_mask[:] = True
        else:
            current_mask[list(current_safe_actions)] = True
            if not current_mask.any():
                raise ValueError("At least one safe current action is required")

        self.memory.append((
            np.array(state, copy=True),
            int(executed_action),
            float(reward),
            np.array(next_state, copy=True),
            bool(done),
            mask,
            current_mask,
        ))

    def set_expert_dataset(self, observations, q_targets):
        observations = torch.as_tensor(observations, dtype=torch.float32)
        q_targets = torch.as_tensor(q_targets, dtype=torch.float32)
        if observations.ndim != 2 or q_targets.shape != (len(observations), 4):
            raise ValueError("Invalid planner-teacher dataset")
        if not len(observations):
            raise ValueError("Empty planner-teacher dataset")
        if not torch.isfinite(observations).all() or not torch.isfinite(q_targets).all():
            raise ValueError("Nonfinite planner-teacher data")
        self.expert_dataset = (observations.clone(), q_targets.clone())

    def pretrain_expert(self, updates, learning_rate=1e-3):
        if not isinstance(updates, int) or updates < 0:
            raise ValueError("expert updates must be nonnegative integer")
        if updates and self.expert_dataset is None:
            raise ValueError("Provide a planner-teacher dataset before pretraining")
        if updates == 0:
            return None
        # A separate optimizer leaves the regular RL learning rate unchanged.
        optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        observations, labels = self.expert_dataset
        for _ in range(updates):
            ids = torch.randint(len(observations), (min(64, len(observations)),))
            loss = F.mse_loss(self.online(observations[ids]), labels[ids])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
            optimizer.step()
        self.target.load_state_dict(self.online.state_dict())
        return float(F.mse_loss(self.online(observations), labels).item())

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

        all_q = self.online(states)
        current = all_q.gather(1, actions).squeeze(1)

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

        td_loss = nn.functional.smooth_l1_loss(current, targets)
        safety_loss = torch.zeros((), dtype=td_loss.dtype)
        if self.safety_loss_weight > 0:
            safe_mask = torch.as_tensor(
                np.stack([t[6] for t in batch]), dtype=torch.bool
            )
            unsafe_mask = ~safe_mask
            if unsafe_mask.any():
                # A supervised ranking constraint on *current* states:
                # every known-unsafe action should score at least margin
                # below the best available safe action. Detaching the
                # reference prevents this auxiliary loss from inflating
                # safe Q-values just to satisfy the margin.
                best_safe_q = all_q.masked_fill(
                    ~safe_mask, -torch.inf
                ).max(dim=1).values.detach()
                violation = torch.relu(
                    all_q - best_safe_q.unsqueeze(1) + self.safety_margin
                )
                safety_loss = (
                    violation.square() * unsafe_mask.float()
                ).sum() / unsafe_mask.float().sum()

        expert_loss = torch.zeros((), dtype=td_loss.dtype)
        if self.expert_loss_weight > 0:
            if self.expert_dataset is None:
                raise RuntimeError("Missing expert dataset for auxiliary loss")
            expert_obs, expert_targets = self.expert_dataset
            ids = torch.randint(len(expert_obs), (min(self.batch_size, len(expert_obs)),))
            expert_loss = F.mse_loss(self.online(expert_obs[ids]), expert_targets[ids])
        loss = (td_loss + self.safety_loss_weight * safety_loss
                + self.expert_loss_weight * expert_loss)

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
            "td_loss": float(td_loss.item()),
            "safety_loss": float(safety_loss.item()),
            "expert_loss": float(expert_loss.item()),
        }

        return float(loss.item())
