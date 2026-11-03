from collections import deque
import random
import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, observation_size=2, action_count=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, observations):
        return self.net(observations)


class DQNAgent:
    def __init__(self, seed=42, gamma=0.99, lr=1e-3, batch_size=64, capacity=10000):
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.online = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = deque(maxlen=capacity)
        self.gamma = gamma
        self.batch_size = batch_size
        self.updates = 0

    def act(self, observation, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(4)
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            return int(self.online(obs).argmax(dim=1).item())

    def remember(self, state, executed_action, reward, next_state, done):
        self.memory.append((state, executed_action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.rng.sample(list(self.memory), self.batch_size)
        states = torch.as_tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
        actions = torch.as_tensor([x[1] for x in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.as_tensor([x[2] for x in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.array([x[3] for x in batch]), dtype=torch.float32)
        dones = torch.as_tensor([x[4] for x in batch], dtype=torch.float32)
        current = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            future = self.target(next_states).max(dim=1).values
            targets = rewards + self.gamma * future * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(current, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
