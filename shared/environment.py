import cv2
import numpy as np
import gym
import random
from shared.config import ENV_NAME, MAX_NOOP_STEPS


class Environment:
    def __init__(self, env_name=ENV_NAME, no_op_steps=MAX_NOOP_STEPS):
        self.env = gym.make(env_name)
        self.no_op_steps = no_op_steps
        self.shape = (84, 84)
        self.state = None

    def process_frame(self, frame):
        frame = frame[34:34 + 160, :160]
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.shape, interpolation=cv2.INTER_NEAREST)
        frame = frame.reshape((*self.shape, 1))
        return frame

    def reset(self, evaluation=False):
        frame = self.env.reset()
        frame = self.process_frame(frame)
        self.state = np.repeat(frame, 4, axis=2)
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)

    def step(self, action, return_original_frame=False):
        frame, reward, terminal, info = self.env.step(action)
        processed_frame = self.process_frame(frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)
        if return_original_frame:
            return frame, reward, Environment.clip_reward(reward), terminal
        return processed_frame, reward, Environment.clip_reward(reward), terminal

    @staticmethod
    def clip_reward(reward):
        if reward > 0:
            return 1
        elif reward == 0:
            return 0
        else:
            return -1
