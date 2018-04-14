import random

import gym
import numpy
from tqdm import tqdm as progress_bar

import tensorflow as tf
from models.neural_network import BreakoutNeuralNetwork
from PIL import Image
from visualization import plot_cost_to_go
from visualization import plot_running_avg


IMG_SIZE = (210, 160)


class Experience:
    def __init__(self, experience_size=100):
        self.experience_size = experience_size
        self.buffer = []

    def sample(self, sample_size=4):
        return random.sample(self.buffer, sample_size)

    def add_sample(self, state, action, reward, new_state):
        state = convert_array(state)
        action = convert_array(action)
        reward = convert_array([reward])
        new_state = convert_array(new_state)
        if len(self.buffer) >= self.experience_size:
            self.buffer.pop()
        self.buffer.insert(0, [state, action, reward, new_state])


def convert_array(observation):
    return numpy.array(observation).reshape((1, -1)).astype(numpy.float32)


def to_state_action_value(model, sample, gamma):
    next_action_predictions = model.predict(sample[3])
    return sample[2] + gamma * numpy.max(next_action_predictions[0])


def map_list_to_array(func, li):
    return numpy.array(list(map(func, li))).reshape((len(li), -1))


def update(model, target_model, experience, gamma, experience_replay, n_actions=2, n_samples=4):
    if len(experience.buffer) >= n_samples and experience_replay:
        samples = experience.sample(n_samples)
    else:
        samples = [experience.buffer[0]]

    if len(experience.buffer) == n_samples or not experience_replay:
        states = map_list_to_array(lambda sample: sample[0], samples)
        state_action_values = numpy.empty((n_samples, n_actions))
        state_action_values[:] = map_list_to_array(
            lambda sample: to_state_action_value(target_model, sample, gamma), samples)
        model.update(states, state_action_values)


def convert_image(observation):
    image = observation[55:-15, 7:-7]
    #image = Image.fromarray(image.astype(numpy.int))
    #image = image.resize((int(IMG_SIZE[0] / 2), int(IMG_SIZE[1] / 2)))
    image = numpy.average(numpy.array(image), axis=2)
    image = image / 255.0
    return convert_array(image)


def frames_to_state(frames_list):
    return numpy.hstack(frames_list)


def update_frame_list(frames, next_observation):
    frames.pop()
    frames.insert(0, next_observation)


def play_one_episode(session, environment, epsilon, gamma=0.99, max_steps=10000, experience_replay=True,
                     use_dual_model=True):
    observation = environment.reset()
    observation = convert_image(observation)
    frames_per_state = 4
    frames = [observation for _ in range(frames_per_state)]
    done = False
    time_step = 0
    total_reward = 0
    experience = Experience()
    dims = observation.shape[1] * frames_per_state
    n_actions = environment.action_space.n
    n_samples = 4
    model = BreakoutNeuralNetwork(session, dims, n_actions, environment)
    if use_dual_model:
        target_model = BreakoutNeuralNetwork(session, dims, n_actions, environment)
    else:
        target_model = model
    session.run(tf.global_variables_initializer())
    while not done and time_step < max_steps:
        time_step += 1
        # Choose E-greedy action
        state = frames_to_state(frames)
        action = model.sample_action(state, epsilon)
        # Take action, observe
        next_observation, reward, done, info = environment.step(action)
        next_observation = convert_image(next_observation)
        # Update state and get new state
        update_frame_list(frames, next_observation)
        next_state = frames_to_state(frames)
        # Adjust reward if episode ended
        total_reward += reward
        # Save experience
        experience.add_sample(state, action, reward, next_state)
        # Update
        update(model, target_model, experience, gamma, experience_replay, n_actions=n_actions, n_samples=n_samples)
        # Update dual network
        if use_dual_model and time_step % 100 == 0:
            update(target_model, target_model,  experience, gamma, experience_replay, n_actions=n_actions, n_samples=n_samples)
        if done:
            break
    return total_reward


def play_multiple_episodes(environment, episodes, experience_replay, use_dual_model):
    with tf.Session() as session:
        total_rewards = numpy.empty(episodes)
        for i in progress_bar(range(episodes), desc='Playing episode'):
            epsilon = 1.0 / numpy.sqrt(1 + i)
            total_rewards[i] = play_one_episode(session, environment, epsilon, experience_replay, use_dual_model)
        plot_running_avg(total_rewards)
        plot_cost_to_go(environment.observation_space)


if __name__ == '__main__':
    play_multiple_episodes(gym.make('Breakout-v0'), 300, experience_replay=False, use_dual_model=True)
