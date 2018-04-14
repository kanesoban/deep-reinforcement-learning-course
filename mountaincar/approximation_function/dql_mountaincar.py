import random

import gym
import numpy
from tqdm import tqdm as progress_bar

import tensorflow as tf
from models.neural_network import MountainCarNeuralNetwork
from visualization import plot_cost_to_go
from visualization import plot_running_avg


class Experience:
    def __init__(self, experience_size=100):
        self.experience_size = experience_size
        self.buffer = []

    def sample(self, sample_size=4):
        return random.sample(self.buffer, sample_size)

    def add_sample(self, state, action, reward, new_state):
        state = numpy.array(state).reshape((1, -1)).astype(numpy.float32)
        action = numpy.array(action).reshape((1, -1)).astype(numpy.float32)
        reward = numpy.array([reward]).reshape((1, -1)).astype(numpy.float32)
        new_state = numpy.array(new_state).reshape((1, -1)).astype(numpy.float32)
        if len(self.buffer) >= self.experience_size:
            self.buffer.pop()
        self.buffer.insert(0, [state, action, reward, new_state])


def convert_observation(observation):
    return numpy.array(observation).reshape((1, -1)).astype(numpy.float32)


def to_state_action_value(model, sample, gamma):
    next_action_predictions = model.predict(sample[3])
    return sample[2] + gamma * numpy.max(next_action_predictions[0])


def update(model, experience, gamma, experience_replay, n_actions=2, n_samples=4):
    if len(experience.buffer) >= n_samples and experience_replay:
        samples = experience.sample(n_samples)
    else:
        samples = [experience.buffer[0]]

    if len(experience.buffer) >= n_samples or not experience_replay:
        states = numpy.array(list(map(lambda sample: sample[0], samples))).reshape((n_samples, -1))
        state_action_values = numpy.array(list(map(lambda sample: to_state_action_value(model, sample, gamma),
                                                   samples))).reshape((n_samples,))
        state_action_values2 = numpy.empty((n_samples, n_actions))
        for i in range(n_actions):
            state_action_values2[:, i] = state_action_values
        model.update(states, state_action_values2)


def play_one_episode(session, environment, epsilon, gamma=0.99, max_steps=10000, experience_replay=True):
    observation = environment.reset()
    done = False
    time_step = 0
    total_reward = 0
    experience = Experience()

    n_actions = 2
    n_samples = 4
    model = MountainCarNeuralNetwork(session, n_actions, environment)
    session.run(tf.global_variables_initializer())
    while not done and time_step < max_steps:
        time_step += 1
        # Choose E-greedy action
        action = model.sample_action(convert_observation(observation), epsilon)
        # Take action, observe
        next_observation, reward, done, info = environment.step(action)
        # Adjust reward if episode ended
        total_reward += reward
        if done:
            reward = 100
        # Save experience
        experience.add_sample(observation, action, reward, next_observation)
        # Update
        update(model, experience, gamma, experience_replay, n_actions=n_actions, n_samples=n_samples)
        if done:
            break
        observation = next_observation
    return total_reward


def play_multiple_episodes(environment, episodes, experience_replay):
    with tf.Session() as session:
        total_rewards = numpy.empty(episodes)
        for i in progress_bar(range(episodes), desc='Playing episode'):
            epsilon = 1.0 / numpy.sqrt(1 + i)
            total_rewards[i] = play_one_episode(session, environment, epsilon, experience_replay)
        plot_running_avg(total_rewards)
        plot_cost_to_go(environment.observation_space)


if __name__ == '__main__':
    play_multiple_episodes(gym.make('MountainCar-v0'), 300, experience_replay=False)
