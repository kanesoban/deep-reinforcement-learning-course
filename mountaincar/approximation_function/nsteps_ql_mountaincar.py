import gym
import numpy
from tqdm import tqdm as progress_bar

from feature_transformers import MountainCarRBFFeatureTransformer
from models import SGDModel
from visualization import plot_cost_to_go
from visualization import plot_running_avg


def calculate_target(rewards, state_action_values):
    return rewards[-1] + sum(state_action_values[:-1])


def update_nstep_rewards(reward, rewards, gamma, time_step, n_step):
    rewards = [e * gamma for e in rewards]
    if time_step >= n_step:
        rewards = rewards[1:] + [reward]
    else:
        rewards.append(reward)
    return rewards


def update_nstep_state_action_values(state_action_value, state_action_values, gamma, time_step, n_step):
    state_action_values = [e * gamma for e in state_action_values]
    if time_step >= n_step:
        state_action_values = state_action_values[1:] + [state_action_value]
    else:
        state_action_values.append(state_action_value)
    return state_action_values


def play_one_episode(environment, epsilon, n_step=1, gamma=0.99, max_steps=10000):
    observation = environment.reset()
    done = False
    time_step = 0
    total_reward = 0
    sampler = environment.observation_space
    feature_transformer = MountainCarRBFFeatureTransformer(sampler)
    model = SGDModel(environment, feature_transformer)
    rewards = []
    state_action_values = []
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
        rewards = update_nstep_rewards(reward, rewards, gamma, time_step, n_step)
        state_action_values = update_nstep_state_action_values(numpy.max(next_action_predictions[0]),
                                                               state_action_values, gamma, time_step, n_step)
        target = calculate_target(rewards, state_action_values)
        model.update(observation, action, target)
        if done:
            break
        observation = next_observation
    return total_reward


def play_multiple_episodes(environment, episodes):
    total_rewards = numpy.empty(episodes)
    for i in progress_bar(range(episodes), desc='Playing episode'):
        epsilon = 1.0 / numpy.sqrt(1 + i)
        total_rewards[i] = play_one_episode(environment, epsilon, n_step=3)
    plot_running_avg(total_rewards)
    plot_cost_to_go(environment.observation_space)


if __name__ == '__main__':
    play_multiple_episodes(gym.make('MountainCar-v0'), 300)
