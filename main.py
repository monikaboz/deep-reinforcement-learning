import numpy as np
import tensorflow as tf
from agents.dddqn_agent import DDDQNAgent
from shared.config import (SAVE_PATH, WRITE_TENSORBOARD, TENSORBOARD_PATH, TOTAL_FRAMES, MAX_EPISODE_LENGTH,
                           UPDATE_FREQ, MIN_REPLAY_BUFFER_SIZE, TARGET_UPDATE_FREQ, EVALUATION_FREQ, EVALUATION_LENGTH)
from shared.environment import Environment
from shared.replay_buffer import ReplayBuffer

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

if __name__ == "__main__":
    env = Environment()
    replay_buffer = ReplayBuffer()
    agent = DDDQNAgent(env.env.action_space.n, replay_buffer)
    writer = tf.summary.create_file_writer(TENSORBOARD_PATH)

    frame_number = 0
    rewards = []
    losses = []
    with writer.as_default():
        while frame_number < TOTAL_FRAMES:
            epoch_frame = 0
            while epoch_frame < EVALUATION_FREQ:
                env.reset()
                episode_reward_sum = 0
                for _ in range(MAX_EPISODE_LENGTH):
                    action = agent.get_action(frame_number, env.state)
                    processed_frame, reward, terminal = env.step(action)
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward
                    agent.add_experience(processed_frame[:, :, 0], action, reward, terminal)

                    if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                        loss, _ = agent.train()
                        losses.append(loss)

                    if frame_number % TARGET_UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                        agent.update_target_model()

                    if terminal:
                        break

                rewards.append(episode_reward_sum)

                if WRITE_TENSORBOARD:
                    tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                    tf.summary.scalar('Loss', np.mean(losses[-100:]), frame_number)
                    writer.flush()
                print(f'Average reward: {np.mean(rewards[-10:]):0.1f} @ Episode number: {str(len(rewards)).zfill(6)} '
                      f'Frame number: {str(frame_number).zfill(8)}')

            terminal = True
            evaluation_rewards = []
            evaluation_frame_number = 0
            for _ in range(EVALUATION_LENGTH):
                if terminal:
                    env.reset(evaluation=True)
                    episode_reward_sum = 0
                    terminal = False

                action = agent.get_action(frame_number, env.state, evaluation=True)
                _, reward, terminal = env.step(action)
                evaluation_frame_number += 1
                episode_reward_sum += reward

                if terminal:
                    evaluation_rewards.append(episode_reward_sum)

            if len(evaluation_rewards) > 0:
                evaluation_avg_reward = np.mean(evaluation_rewards)
            else:
                evaluation_avg_reward = episode_reward_sum

            if WRITE_TENSORBOARD:
                tf.summary.scalar('Evaluation Reward', evaluation_avg_reward, frame_number)
                writer.flush()
            print(f'Average evaluation reward: {np.mean(rewards[-10:]):0.1f}')

            if SAVE_PATH is not None:
                agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}')
