"""Simple DQN: executed actions are stored in replay; original proposals logged separately."""
from collections import deque
import random
import numpy as np
import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, seed=42, lr=1e-3, gamma=0.99, batch_size=64, capacity=10000):
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.online, self.target = QNetwork(), QNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = deque(maxlen=capacity)
        self.batch_size, self.gamma, self.updates = batch_size, gamma, 0

    def q_values(self, observation):
        with torch.no_grad():
            return self.online(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))[0].numpy()

    def act(self, observation, epsilon=0.0):
        return self.rng.randrange(4) if self.rng.random() < epsilon else int(np.argmax(self.q_values(observation)))

    def remember(self, state, executed_action, reward, next_state, done):
        self.memory.append((np.array(state, copy=True), int(executed_action), float(reward),
                            np.array(next_state, copy=True), bool(done)))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.rng.sample(self.memory, self.batch_size)
        states = torch.as_tensor(np.stack([t[0] for t in batch]), dtype=torch.float32)
        actions = torch.as_tensor([t[1] for t in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.as_tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([t[3] for t in batch]), dtype=torch.float32)
        dones = torch.as_tensor([t[4] for t in batch], dtype=torch.float32)
        current = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            targets = rewards + self.gamma * self.target(next_states).max(1).values * (1 - dones)
        loss = nn.functional.smooth_l1_loss(current, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
