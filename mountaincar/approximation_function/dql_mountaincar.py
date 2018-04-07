import gym
import numpy
from tqdm import tqdm as progress_bar

import tensorflow as tf
from models.neural_network import MountainCarNeuralNetwork
from visualization import plot_cost_to_go
from visualization import plot_running_avg


def play_one_episode(environment, epsilon, gamma=0.99, max_steps=10000):
    observation = environment.reset()
    done = False
    time_step = 0
    total_reward = 0
    session = tf.Session()
    model = MountainCarNeuralNetwork(session, 4, 2)

    while not done and time_step < max_steps:
        time_step += 1
        # Choose E-greedy action
        action = model.sample_action(observation, epsilon)
        # Take action, observe
        next_observation, reward, done, info = environment.step(action)
        # Adjust reward if episode ended
        total_reward += reward
        if done:
            reward = 100
        # Update
        next_action_predictions = model.predict(next_observation)
        state_action_value = reward + gamma * numpy.max(next_action_predictions[0])
        model.update(observation, state_action_value)
        if done:
            break
        observation = next_observation
    return total_reward


def play_multiple_episodes(environment, episodes):
    total_rewards = numpy.empty(episodes)
    for i in progress_bar(range(episodes), desc='Playing episode'):
        epsilon = 1.0 / numpy.sqrt(1 + i)
        total_rewards[i] = play_one_episode(environment, epsilon)
    plot_running_avg(total_rewards)
    plot_cost_to_go(environment.observation_space)


if __name__ == '__main__':
    play_multiple_episodes(gym.make('MountainCar-v0'), 300)
