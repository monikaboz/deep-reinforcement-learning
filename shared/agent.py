import os
from abc import ABC, abstractmethod
from os import path
import numpy as np
from tensorflow.keras.models import load_model
from shared.config import (LEARNING_RATE, BATCH_SIZE, MIN_REPLAY_BUFFER_SIZE, TOTAL_FRAMES, EPS_INITIAL, EPS_FINAL,
                           EPS_ANNEALING_FRAMES, EPS_FINAL_FRAME, EPS_EVALUATION, TARGET_LEARNING_RATE)


class Agent(ABC):
    def __init__(self, n_actions, replay_buffer, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                 min_replay_buffer_size=MIN_REPLAY_BUFFER_SIZE, total_frames=TOTAL_FRAMES, eps_initial=EPS_INITIAL,
                 eps_final=EPS_FINAL, eps_annealing_frames=EPS_ANNEALING_FRAMES, eps_final_frame=EPS_FINAL_FRAME,
                 eps_evaluation=EPS_EVALUATION):
        self.n_actions = n_actions
        self.replay_buffer = replay_buffer

        self.model = self.build_model(learning_rate)
        self.target_model = self.build_target_model()

        self.batch_size = batch_size
        self.min_replay_buffer_size = min_replay_buffer_size
        self.total_frames = total_frames

        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames

        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.min_replay_buffer_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.total_frames - self.eps_annealing_frames -
                                                                   self.min_replay_buffer_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.total_frames

    def calc_epsilon(self, frame_number, evaluation=False):
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.min_replay_buffer_size:
            return self.eps_initial
        elif self.min_replay_buffer_size <= frame_number < self.min_replay_buffer_size + self.eps_annealing_frames:
            return self.slope * frame_number + self.intercept
        elif frame_number >= self.min_replay_buffer_size + self.eps_annealing_frames:
            return self.slope_2 * frame_number + self.intercept_2

    @abstractmethod
    def build_model(self, learning_rate):
        pass

    @abstractmethod
    def train(self, discount_factor):
        pass

    @abstractmethod
    def build_target_model(self, learning_rate):
        pass

    def update_target_model(self):
        if self.target_model is not None:
            self.target_model.set_weights(self.model.get_weights())

    def get_action(self, frame_number, state, evaluation=False):
        eps = self.calc_epsilon(frame_number, evaluation)
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)
        q_values = self.model.predict(state.reshape((-1, 84, 84, 4)))[0]
        return q_values.argmax()

    def add_experience(self, frame, action, reward, terminal):
        self.replay_buffer.add(frame, action, reward, terminal)

    def save(self, folder_name):
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        self.model.save(folder_name + '/dqn.h5')
        if self.target_model is not None:
            self.target_model.save(folder_name + '/target_dqn.h5')

    def load(self, folder_name):
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        self.model = load_model(folder_name + '/dqn.h5')
        target_dqn_path = folder_name + '/target_dqn.h5'
        if path.exists(target_dqn_path):
            self.target_model = load_model(target_dqn_path)
