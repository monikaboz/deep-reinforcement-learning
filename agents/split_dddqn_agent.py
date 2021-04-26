import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from shared.agent import Agent
from shared.config import DISCOUNT_FACTOR, TARGET_LEARNING_RATE


class SplitDDDQNAgent(Agent):
    def build_model(self, learning_rate):
        input = Input(shape=(84, 84, 4))
        val_x = Lambda(lambda layer: layer / 255)(input)
        val_x = Conv2D(32, 8, 4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(val_x)
        val_x = Conv2D(64, 4, 2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(val_x)
        val_x = Conv2D(64, 3, 1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(val_x)
        val_x = Conv2D(512, 7, 1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
            val_x)
        val_x = Flatten()(val_x)
        val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_x)

        diff_input = Input(shape=(84, 84, 4))
        adv_x = Lambda(lambda layer: layer / 255)(diff_input)
        adv_x = Conv2D(32, 8, 4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(adv_x)
        adv_x = Conv2D(64, 4, 2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(adv_x)
        adv_x = Conv2D(64, 3, 1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(adv_x)
        adv_x = Conv2D(512, 7, 1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
            adv_x)
        adv_x = Flatten()(adv_x)
        adv = Dense(self.n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_x)

        reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))
        q_values = Add()([val, Subtract()([adv, reduce_mean(adv)])])

        model = Model([input, diff_input], q_values)
        model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
        return model

    def train(self, discount_factor=DISCOUNT_FACTOR):
        states, actions, rewards, next_states, terminals = self.replay_buffer.get_minibatch(self.batch_size)
        next_states, diff_next_states = SplitDDDQNAgent.split_states(next_states)
        arg_q_max = self.model.predict([next_states, diff_next_states]).argmax(axis=1)
        next_q_values = self.target_model.predict([next_states, diff_next_states])
        double_q = next_q_values[range(self.batch_size), arg_q_max]
        target_q = rewards + (discount_factor * double_q * (1 - terminals))

        with tf.GradientTape() as tape:
            states, diff_states = SplitDDDQNAgent.split_states(states)
            q_values = self.model([states, diff_states])
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions, dtype=np.float32)
            q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            loss = tf.keras.losses.Huber()(target_q, q)
            error = q - target_q

        model_gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(model_gradients, self.model.trainable_variables))
        return float(loss.numpy()), error

    def build_target_model(self, learning_rate=TARGET_LEARNING_RATE):
        return self.build_model(learning_rate)

    def get_action(self, frame_number, state, evaluation=False):
        eps = self.calc_epsilon(frame_number, evaluation)
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)
        states, diff_states = SplitDDDQNAgent.split_states(state.reshape((-1, 84, 84, 4)))
        q_values = self.model.predict([states, diff_states])[0]
        return q_values.argmax()

    @staticmethod
    def split_states(state):
        diff_state = state[:, :, :, -1, np.newaxis]
        for i in range(state.shape[3] - 1, 0, -1):
            diff_state = np.append(diff_state, state[:, :, :, i, np.newaxis] - state[:, :, :, i - 1, np.newaxis],
                                   axis=3)
        return state, diff_state
