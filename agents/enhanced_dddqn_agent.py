import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from shared.agent import Agent
from shared.config import DISCOUNT_FACTOR, TARGET_LEARNING_RATE


class EnhancedDDDQNAgent(Agent):
    def build_model(self, learning_rate):
        input = Input(shape=(84, 84, 4))
        normalized_input = Lambda(lambda layer: layer / 255)(input)

        last_frame_input = Lambda(lambda layer: tf.slice(layer, [0, 0, 0, 3], [-1, -1, -1, 1]))(normalized_input)
        first_frame_input = Lambda(lambda layer: tf.slice(layer, [0, 0, 0, 0], [-1, -1, -1, 1]))(normalized_input)
        diff_frame_input = Subtract()([last_frame_input, first_frame_input])

        x = Conv2D(32, 8, 4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
            normalized_input)
        x = Conv2D(64, 4, 2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
        x = Conv2D(64, 3, 1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
        x = Conv2D(1024, 7, 1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

        val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)

        val_x = Dense(512, kernel_initializer=VarianceScaling(scale=2.))(last_frame_input)
        val_stream = Concatenate()([Flatten()(val_stream), Flatten()(val_x)])
        val_stream = Flatten()(val_stream)
        val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

        adv_x = Dense(512, kernel_initializer=VarianceScaling(scale=2.))(diff_frame_input)
        adv_stream = Concatenate()([Flatten()(adv_stream), Flatten()(adv_x)])
        adv_stream = Flatten()(adv_stream)
        adv = Dense(self.n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

        reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))
        q_values = Add()([val, Subtract()([adv, reduce_mean(adv)])])

        model = Model(input, q_values)
        model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
        return model

    def train(self, discount_factor=DISCOUNT_FACTOR):
        states, actions, rewards, next_states, terminals = self.replay_buffer.get_minibatch(self.batch_size)
        arg_q_max = self.model.predict(next_states).argmax(axis=1)
        next_q_values = self.target_model.predict(next_states)
        double_q = next_q_values[range(self.batch_size), arg_q_max]
        target_q = rewards + (discount_factor * double_q * (1 - terminals))

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions, dtype=np.float32)
            q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            loss = tf.keras.losses.Huber()(target_q, q)
            error = q - target_q

        model_gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(model_gradients, self.model.trainable_variables))
        return float(loss.numpy()), error

    def build_target_model(self, learning_rate=TARGET_LEARNING_RATE):
        return self.build_model(learning_rate)
