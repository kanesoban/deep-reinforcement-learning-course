import gym
import numpy
from tqdm import tqdm as progress_bar

from feature_transformers import MountainCarCompoundRBFFeatureTransformer
from feature_transformers import MountainCarRBFFeatureTransformer
from models.numpy import ActorCriticModel
from models.numpy import SGDStateModel
from samplers import CompoundSampler
from visualization import plot_cost_to_go
from visualization import plot_running_avg


def play_one_episode(environment, epsilon, gamma=0.99, max_steps=10000):
    observation = environment.reset()
    done = False
    time_step = 0
    total_reward = 0
    state_sampler = environment.observation_space
    compound_sampler = CompoundSampler(environment, numpy.array([0, 1]))
    state_feature_transformer = MountainCarRBFFeatureTransformer(state_sampler)
    compound_feature_transformer = MountainCarCompoundRBFFeatureTransformer(compound_sampler)
    state_value_model = SGDStateModel(environment, state_feature_transformer)
    policy_model = ActorCriticModel(environment, compound_feature_transformer)
    while not done and time_step < max_steps:
        time_step += 1
        # Choose action via softmax
        action = policy_model.sample_action(observation, epsilon)
        # Take action, observe
        next_observation, reward, done, info = environment.step(action)
        # Adjust reward if episode ended
        total_reward += reward
        if done:
            reward = 100
        # Update
        state_value = reward + gamma * state_value_model.predict(next_observation)
        policy_model.update(observation, action, state_value)
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
