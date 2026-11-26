"""Double DQN agent with an optional shield-safe bootstrap target."""
from collections import deque
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    """Plain MLP on the raw observation."""

    def __init__(self, input_dim=104):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)


class SpatialQNetwork(nn.Module):
    """MLP with features computed from the map observation.

    Adds one-hot agent and goal planes, the 5x5 wall patch around the agent
    and the goal-relative offset. These are deterministic functions of the
    observation (position, goal, occupancy grid); nothing is masked.
    """

    def __init__(self, input_dim=104):
        super().__init__()
        cells = input_dim - 4
        size = int(cells ** .5)
        if size < 2 or size * size != cells:
            raise ValueError("spatial network needs a map observation (4 + square grid)")
        self.size = size
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2 * cells + 27, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, observation):
        one = observation.ndim == 1
        x = observation.unsqueeze(0) if one else observation
        b, n = x.shape
        if n != self.input_dim:
            raise ValueError(f"expected {self.input_dim} observation features, got {n}")
        size = self.size

        def cell(i):
            return (x[:, i] * (size - 1)).round().long().clamp(0, size - 1)

        row, col, goal_row, goal_col = cell(0), cell(1), cell(2), cell(3)
        agent_plane = F.one_hot(row * size + col, size * size).to(x.dtype)
        goal_plane = F.one_hot(goal_row * size + goal_col, size * size).to(x.dtype)
        walls = F.pad(x[:, 4:].reshape(b, 1, size, size), (2, 2, 2, 2), value=1.0)
        offsets = torch.arange(5, device=x.device)
        patch = ((row[:, None, None] + offsets[None, :, None]) * (size + 4)
                 + (col[:, None, None] + offsets[None, None, :])).reshape(b, 25)
        local_walls = walls.reshape(b, -1).gather(1, patch)
        relative_goal = torch.stack((goal_row - row, goal_col - col), dim=1).to(x.dtype) / (size - 1)
        values = self.net(torch.cat((x, agent_plane, goal_plane, local_walls, relative_goal), dim=1))
        return values[0] if one else values


def create_network(input_dim, architecture="spatial"):
    if architecture == "mlp":
        return QNetwork(input_dim)
    if architecture == "spatial":
        return SpatialQNetwork(input_dim)
    raise ValueError(f"Unknown architecture: {architecture}")


def network_from_weights(weights):
    """Rebuild the right network type from a saved state_dict."""
    dim = int(weights["net.0.weight"].shape[1])
    area = (dim - 31) // 3  # spatial first layer has 31 + 3 * area inputs
    side = int(area ** .5) if area > 0 else 0
    if dim >= 43 and (dim - 31) % 3 == 0 and side * side == area:
        model = create_network(4 + area, "spatial")
    else:
        model = create_network(dim, "mlp")
    model.load_state_dict(weights)
    model.eval()
    return model


class DQNAgent:
    """Double DQN. With backup_mode='shield_safe' the bootstrap action is chosen
    only among actions the shield allows in the next state."""

    def __init__(self, input_dim, seed=42, lr=1e-4, gamma=0.99, batch_size=64,
                 capacity=10000, target_sync=100, backup_mode="standard", architecture="spatial"):
        if backup_mode not in ("standard", "shield_safe"):
            raise ValueError("backup_mode must be 'standard' or 'shield_safe'")
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.online = create_network(input_dim, architecture)
        self.target = create_network(input_dim, architecture)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_sync = target_sync
        self.backup_mode = backup_mode
        self.updates = 0
        self.last_loss = None

    def q_values(self, observation):
        with torch.no_grad():
            return self.online(torch.as_tensor(observation, dtype=torch.float32)).numpy()

    def act(self, observation, epsilon=0.0):
        """Epsilon-greedy proposal (before any shield)."""
        if self.rng.random() < epsilon:
            return self.rng.randrange(4)
        return int(np.argmax(self.q_values(observation)))

    def remember(self, state, action, reward, next_state, done, next_safe_actions=None):
        if self.backup_mode == "shield_safe" and not done and not next_safe_actions:
            raise ValueError("shield_safe backup needs the next state's safe actions")
        mask = np.ones(4, dtype=np.bool_)
        if next_safe_actions is not None:
            mask[:] = False
            mask[list(next_safe_actions)] = True
        self.memory.append((np.array(state, copy=True), int(action), float(reward),
                            np.array(next_state, copy=True), bool(done), mask))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.rng.sample(self.memory, self.batch_size)
        states = torch.as_tensor(np.stack([t[0] for t in batch]), dtype=torch.float32)
        actions = torch.as_tensor([t[1] for t in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.as_tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([t[3] for t in batch]), dtype=torch.float32)
        dones = torch.as_tensor([t[4] for t in batch], dtype=torch.bool)
        masks = torch.as_tensor(np.stack([t[5] for t in batch]), dtype=torch.bool)

        current = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_online = self.online(next_states)  # online net picks the action...
            if self.backup_mode == "shield_safe":
                next_online = next_online.masked_fill(~(masks | dones.unsqueeze(1)), -torch.inf)
            next_actions = next_online.argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions).squeeze(1)  # ...target net values it
            targets = rewards + self.gamma * next_values * (~dones).float()

        loss = F.smooth_l1_loss(current, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite DQN loss")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss
