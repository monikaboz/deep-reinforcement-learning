import os
from datetime import datetime
import imageio
import tensorflow as tf
from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.dddqn_agent import DDDQNAgent
from agents.split_dddqn_agent import SplitDDDQNAgent
from shared.config import ENV_NAME, SAVE_PATH, TOTAL_FRAMES
from shared.environment import Environment
from shared.replay_buffer import ReplayBuffer
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def generate_gif(folder_name, env, agent):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    frames = []
    env.reset()
    while True:
        action = agent.get_action(TOTAL_FRAMES, env.state, True)
        frame, _, _, terminal = env.step(action, True)
        frames.append(frame)
        if terminal:
            break
    imageio.mimsave(f'{folder_name}/{datetime.now()}.gif', frames, duration=1 / 30)


def plot_models(env, replay_buffer):
    # dqn_agent = DQNAgent(env.env.action_space.n, replay_buffer)
    # ddqn_agent = DDQNAgent(env.env.action_space.n, replay_buffer)
    # dddqn_agent = DDDQNAgent(env.env.action_space.n, replay_buffer)
    split_dddqn_agent = SplitDDDQNAgent(env.env.action_space.n, replay_buffer)

    # tf.keras.utils.plot_model(dqn_agent.model, to_file='DQN_model.png', show_shapes=True)
    # tf.keras.utils.plot_model(ddqn_agent.model, to_file='DDQN_model.png', show_shapes=True)
    # tf.keras.utils.plot_model(dddqn_agent.model, to_file='DDDQN_model.png', show_shapes=True)
    tf.keras.utils.plot_model(split_dddqn_agent.model, to_file='split_DDDQN_model.png', show_shapes=True)


def save_frame(frame, name):
    img = Image.fromarray(frame)
    img.save(name)


def save_frame_preprocessing(env):
    frame = env.env.reset()
    for i in range(10):
        frame, _, _, _ = env.env.step(0)
    save_frame(frame, '1_original_frame.png')
    frame = frame[34:34 + 160, :160]
    save_frame(frame, '2_cropped_frame.png')
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    save_frame(frame, '3_gray_frame.png')
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)
    save_frame(frame, '4_resized_frame.png')


def save_exploration_exploation_function(agent):
    epsilons = []
    for i in range(TOTAL_FRAMES):
        epsilons.append(agent.calc_epsilon(i))

    plt.plot(range(TOTAL_FRAMES), epsilons)
    plt.xlabel('Številka slike')
    plt.ylabel('Vrednost ε')
    plt.savefig("epsilon.png")


env = Environment()
replay_buffer = ReplayBuffer()
agent = SplitDDDQNAgent(env.env.action_space.n, replay_buffer)

# agent.model.summary()
# agent.load(SAVE_PATH + 'save-03038839')

# generate_gif(f'{ENV_NAME}-gif/', env, agent)

# plot_models(env, replay_buffer)

save_frame_preprocessing(env)

# save_exploration_exploation_function(agent)

# ! tensorboard --logdir {TENSORBOARD_PATH}




