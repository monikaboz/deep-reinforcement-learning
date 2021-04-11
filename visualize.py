import os
from datetime import datetime
import imageio
import tensorflow as tf
from agents.dddqn_agent import DDDQNAgent
from shared.config import ENV_NAME, SAVE_PATH, TOTAL_FRAMES
from shared.environment import Environment
from shared.replay_buffer import ReplayBuffer

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def generate_gif(folder_name, env, agent):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    frames = []
    env.reset()
    while True:
        action = agent.get_action(TOTAL_FRAMES, env.state, True)
        frame, _, terminal = env.step(action, True)
        frames.append(frame)
        if terminal:
            break
    imageio.mimsave(f'{folder_name}/{datetime.now()}.gif', frames, duration=1 / 30)


env = Environment()
replay_buffer = ReplayBuffer()
agent = DDDQNAgent(env.env.action_space.n, replay_buffer)
agent.load(SAVE_PATH + 'save-00047330')
generate_gif(f'{ENV_NAME}-gif/', env, agent)
# tensorboard --logdir {TENSORBOARD_PATH}
