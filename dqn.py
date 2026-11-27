"""Double DQN agent with an optional shield-safe bootstrap target and margin loss."""
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


class ConvQNetwork(nn.Module):
    """Convolutional Q-network on the map observation.

    The observation (agent, goal, wall grid) becomes three 10x10 planes: walls, agent (one-hot)
    and goal (one-hot). The wall plane is padded with walls (value 1) so the grid edge looks like
    a wall, then two 3x3 convolutions and an MLP head give the 4 action values. Nothing is masked.
    """

    def __init__(self, input_dim=104, channels=(16, 32), hidden=128):
        super().__init__()
        cells = input_dim - 4
        size = int(cells ** .5)
        if size < 2 or size * size != cells:
            raise ValueError("conv network needs a map observation (4 + square grid)")
        self.size, self.input_dim = size, input_dim
        c1, c2 = channels
        self.conv = nn.Sequential(nn.Conv2d(3, c1, 3), nn.ReLU(),            # (size+2)^2 -> size^2
                                  nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(c2 * cells, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def planes(self, x):
        b, size = x.shape[0], self.size

        def cell(i):
            return (x[:, i] * (size - 1)).round().long().clamp(0, size - 1)

        agent = F.one_hot(cell(0) * size + cell(1), size * size).to(x.dtype).reshape(b, 1, size, size)
        goal = F.one_hot(cell(2) * size + cell(3), size * size).to(x.dtype).reshape(b, 1, size, size)
        walls = x[:, 4:].reshape(b, 1, size, size)
        return torch.cat((F.pad(walls, (1, 1, 1, 1), value=1.0), F.pad(agent, (1, 1, 1, 1)),
                          F.pad(goal, (1, 1, 1, 1))), dim=1)

    def forward(self, observation):
        one = observation.ndim == 1
        x = observation.unsqueeze(0) if one else observation
        if x.shape[1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} observation features, got {x.shape[1]}")
        values = self.head(self.conv(self.planes(x)).flatten(1))
        return values[0] if one else values


ARCHITECTURES = ("spatial", "conv", "mlp")


def create_network(input_dim, architecture="spatial"):
    if architecture == "mlp":
        return QNetwork(input_dim)
    if architecture == "spatial":
        return SpatialQNetwork(input_dim)
    if architecture == "conv":
        return ConvQNetwork(input_dim)
    raise ValueError(f"Unknown architecture: {architecture}")


def network_from_weights(weights):
    """Rebuild the right network type from a saved state_dict."""
    if "conv.0.weight" in weights:
        c1, c2 = int(weights["conv.0.weight"].shape[0]), int(weights["conv.2.weight"].shape[0])
        cells = int(weights["head.0.weight"].shape[1]) // c2
        model = ConvQNetwork(4 + cells, (c1, c2), int(weights["head.0.weight"].shape[0]))
        model.load_state_dict(weights)
        model.eval()
        return model
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


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions, stored in numpy arrays.

    ``sample`` draws logical indices (0 = oldest) with ``rng.sample(range(len), k)``.
    That is the same draw ``rng.sample(deque, k)`` makes, so batches match a
    ``deque(maxlen=capacity)`` buffer exactly.
    """

    def __init__(self, capacity, state_dim):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity, self.size, self.oldest = int(capacity), 0, 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.next_masks = np.ones((capacity, 4), dtype=np.bool_)
        self.proposed = np.zeros(capacity, dtype=np.int64)
        self.intervened = np.zeros(capacity, dtype=np.bool_)
        self.safe_masks = np.ones((capacity, 4), dtype=np.bool_)

    def __len__(self):
        return self.size

    def append(self, state, action, reward, next_state, done, next_mask, discount,
               proposed, intervened, safe_mask):
        if self.size < self.capacity:
            i = self.size
            self.size += 1
        else:  # overwrite the oldest entry, like deque(maxlen=...)
            i = self.oldest
            self.oldest = (self.oldest + 1) % self.capacity
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.dones[i], self.next_masks[i] = next_state, done, next_mask
        self.discounts[i], self.proposed[i] = discount, proposed
        self.intervened[i], self.safe_masks[i] = intervened, safe_mask

    def sample(self, rng, k):
        logical = np.asarray(rng.sample(range(self.size), k), dtype=np.int64)
        i = (logical + self.oldest) % self.capacity
        return {name: torch.as_tensor(getattr(self, name)[i]) for name in (
            "states", "actions", "rewards", "next_states", "dones", "next_masks",
            "discounts", "proposed", "intervened", "safe_masks")}


def action_mask(actions):
    """Boolean mask of length 4; None means every action is allowed."""
    mask = np.ones(4, dtype=np.bool_)
    if actions is not None:
        mask[:] = False
        mask[list(actions)] = True
    return mask


def margin_loss(q_all, proposed, safe_mask, intervened, margin, kind="hinge", stop_grad=False):
    """Large-margin loss on blocked proposals (FABR v2; cf. DQfD, Hester et al. 2018).

    For each intervened sample, d = Q(s, a_prop) - (max_{a in safe(s)} Q(s, a) - margin):
        kind="hinge"         relu(d): zero once the blocked action sits `margin` below the
                             best safe action, so the loss switches itself off
        kind="fixed_target"  d ** 2: bounded and two-sided; keeps pulling Q(s, a_prop) to
                             exactly `margin` below the best safe action (ablation)
    Returns the mean over intervened samples (0 if there are none). With stop_grad the
    max over safe actions is treated as a constant.
    """
    if kind not in ("hinge", "fixed_target"):
        raise ValueError(f"Unknown margin loss kind: {kind}")
    if not bool(intervened.any()):
        return q_all.new_zeros(())
    q = q_all[intervened]
    q_prop = q.gather(1, proposed[intervened].unsqueeze(1)).squeeze(1)
    best_safe = q.masked_fill(~safe_mask[intervened], -torch.inf).max(dim=1).values
    if stop_grad:
        best_safe = best_safe.detach()
    d = q_prop - (best_safe - margin)
    return (F.relu(d) if kind == "hinge" else d.pow(2)).mean()


class DQNAgent:
    """Double DQN. With backup_mode='shield_safe' the bootstrap action is chosen
    only among actions the shield allows in the next state.

    ``margin_config`` = dict(kind, margin, weight, stop_grad) adds
    ``weight * margin_loss(...)`` to the TD loss (FABR v2); None disables it.
    """

    def __init__(self, input_dim, seed=42, lr=1e-4, gamma=0.99, batch_size=64,
                 capacity=10000, target_sync=100, backup_mode="standard", architecture="spatial",
                 margin_config=None):
        if backup_mode not in ("standard", "shield_safe"):
            raise ValueError("backup_mode must be 'standard' or 'shield_safe'")
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.online = create_network(input_dim, architecture)
        self.target = create_network(input_dim, architecture)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity, input_dim)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_sync = target_sync
        self.backup_mode = backup_mode
        self.margin_config = dict(margin_config) if margin_config else None
        if self.margin_config:
            margin_loss(torch.zeros(1, 4), torch.zeros(1, dtype=torch.int64),
                        torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, dtype=torch.bool),
                        self.margin_config["margin"], self.margin_config["kind"])  # validates kind
        self.updates = 0
        self.last_loss = None
        self.last_margin_loss = None

    def q_values(self, observation):
        with torch.no_grad():
            return self.online(torch.as_tensor(observation, dtype=torch.float32)).numpy()

    def act(self, observation, epsilon=0.0):
        """Epsilon-greedy proposal (before any shield)."""
        return self.act_with_q(observation, epsilon)[0]

    def act_with_q(self, observation, epsilon=0.0):
        """(epsilon-greedy proposal, Q-values), with a single forward pass."""
        q = self.q_values(observation)
        if self.rng.random() < epsilon:
            return self.rng.randrange(4), q
        return int(np.argmax(q)), q

    def remember(self, state, action, reward, next_state, done, next_safe_actions=None,
                 discount=None, proposed=None, intervened=False, safe_actions=None):
        """Store one transition. ``discount`` multiplies the bootstrap value (gamma ** n for an
        n-step return; default gamma). ``proposed``/``intervened``/``safe_actions`` describe the
        state s and are only used by the margin loss."""
        if self.backup_mode == "shield_safe" and not done and not next_safe_actions:
            raise ValueError("shield_safe backup needs the next state's safe actions")
        if intervened and (proposed is None or safe_actions is None):
            raise ValueError("an intervened transition needs its proposal and safe actions")
        self.memory.append(np.asarray(state, dtype=np.float32), int(action), float(reward),
                           np.asarray(next_state, dtype=np.float32), bool(done),
                           action_mask(next_safe_actions),
                           self.gamma if discount is None else float(discount),
                           int(action if proposed is None else proposed), bool(intervened),
                           action_mask(safe_actions))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        b = self.memory.sample(self.rng, self.batch_size)
        states, rewards, next_states, dones = b["states"], b["rewards"], b["next_states"], b["dones"]
        masks = b["next_masks"]

        q_all = self.online(states)
        current = q_all.gather(1, b["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = self.online(next_states)  # online net picks the action...
            if self.backup_mode == "shield_safe":
                next_online = next_online.masked_fill(~(masks | dones.unsqueeze(1)), -torch.inf)
            next_actions = next_online.argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions).squeeze(1)  # ...target net values it
            targets = rewards + b["discounts"] * next_values * (~dones).float()

        loss = F.smooth_l1_loss(current, targets)
        if self.margin_config:
            c = self.margin_config
            extra = margin_loss(q_all, b["proposed"], b["safe_masks"], b["intervened"],
                                c["margin"], c["kind"], c["stop_grad"])
            self.last_margin_loss = float(extra.item())
            loss = loss + c["weight"] * extra
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
