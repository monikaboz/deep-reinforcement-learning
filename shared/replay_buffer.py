import random
import numpy as np
from shared.config import BATCH_SIZE, REPLAY_BUFFER_SIZE


class ReplayBuffer:
    def __init__(self, size=REPLAY_BUFFER_SIZE):
        self.size = size
        self.frames = np.empty((self.size, 84, 84), dtype=np.uint8)
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.terminals = np.empty(self.size, dtype=np.bool)
        self.count = 0
        self.current = 0

    def add(self, frame, action, reward, terminal):
        self.frames[self.current, ...] = frame
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current)
        self.current = (self.current + 1) % self.size

    def get_valid_indices(self, batch_size):
        indices = []
        for _ in range(batch_size):
            while True:
                index = random.randint(4, self.count)
                if index >= self.current >= index - 4:
                    continue
                if self.terminals[index - 4:index].any():
                    continue
                break
            indices.append(index)
        return indices

    def get_minibatch(self, batch_size=BATCH_SIZE):
        states = []
        next_states = []
        indices = self.get_valid_indices(batch_size)
        for idx in indices:
            states.append(self.frames[idx - 4:idx, ...])
            next_states.append(self.frames[idx - 4 + 1:idx + 1, ...])
        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        next_states = np.transpose(np.asarray(next_states), axes=(0, 2, 3, 1))
        return states, self.actions[indices], self.rewards[indices], next_states, self.terminals[indices]
