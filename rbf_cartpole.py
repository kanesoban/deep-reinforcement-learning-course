import gym
import matplotlib.pyplot as plt
import numpy
import numpy.random as random
from tqdm import tqdm as progress_bar

from feature_transformers import CartPoleRBFFeatureTransformer
from models import SGDModel

gamma = 0.99


class CartPoleSampler:
    def __init__(self):
        self.cart_position_interval = (-1.0, 1.0)
        self.cart_velocity_interval = (-1.0, 1.0)
        self.pole_angle_interval = (-1.0, 1.0)
        self.pole_velocity_interval = (-1.0, 1.0)

    def sample(self):
        return (
            random.uniform(self.cart_position_interval[0], self.cart_position_interval[1]),
            random.uniform(self.cart_velocity_interval[0], self.cart_velocity_interval[1]),
            random.uniform(self.pole_angle_interval[0], self.pole_angle_interval[1]),
            random.uniform(self.pole_velocity_interval[0], self.pole_velocity_interval[1])
        )


def play_one_episode(environment, epsilon, sampler):
    observation = environment.reset()
    done = False
    time_step = 0
    feature_transformer = CartPoleRBFFeatureTransformer(sampler)
    model = SGDModel(environment, feature_transformer)
    while not done and time_step < 10000:
        time_step += 1
        # Choose E-greedy action
        action = model.sample_action(observation, epsilon)
        # Take action, observe
        next_observation, reward, done, info = environment.step(action)
        # Adjust reward if episode ended
        if done and time_step < 199:
            reward = -300
        # Update
        next_action_predictions = model.predict(next_observation)
        state_action_value = reward + gamma * numpy.max(next_action_predictions[0])
        model.update(observation, action, state_action_value)
        if done:
            break
        observation = next_observation
    return time_step


def play_multiple_episodes(environment, episodes):
    sampler = CartPoleSampler()
    episode_lengths = numpy.empty(episodes)
    for i in progress_bar(range(episodes), desc='Playing episode'):
        epsilon = 1.0 / numpy.sqrt(1 + i)
        episode_lengths[i] = play_one_episode(environment, epsilon, sampler)
    plt.plot(episode_lengths)
    plt.savefig('episode_lengths.png')


if __name__ == '__main__':
    play_multiple_episodes(gym.make('CartPole-v0'), 300)

