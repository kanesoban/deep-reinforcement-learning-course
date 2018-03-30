from __future__ import print_function, division
from builtins import range
import collections
from operator import itemgetter
import random

import gym
import numpy as np
import matplotlib.pyplot as plt


learning_rate = 1e-2
discount_factor = 0.9
n_bins = 10
cart_pos_bins = np.linspace(-2.4, 2.4, n_bins)
cart_vel_bins = np.linspace(-2, 2, n_bins)
pole_ang_bins = np.linspace(-0.4, 0.4, n_bins)
pole_vel_bins = np.linspace(-3.5, 3.5, n_bins)
Q = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))


def extend_bins(bins, value):
    max_val = np.max(bins)
    while max_val < value:
        max_val += bins[1] - bins[0]
        bins = np.append(bins, max_val)
    min_val = np.min(bins)
    while min_val > value:
        min_val -= bins[1] - bins[0]
        bins = np.insert(bins, 0, min_val)
    return bins


def get_bin(bins, value):
    ind = np.where(bins >= value)[0][0]
    return (bins[ind] + bins[ind-1]) / 2.0


def get_discreet_state(cart_pos, cart_vel, pole_ang, pole_vel):
    global cart_pos_bins
    cart_pos_bins = extend_bins(cart_pos_bins, cart_pos)
    global cart_vel_bins
    cart_vel_bins = extend_bins(cart_vel_bins, cart_vel)
    global pole_ang_bins
    pole_ang_bins = extend_bins(pole_ang_bins, pole_ang)
    global pole_vel_bins
    pole_vel_bins = extend_bins(pole_ang_bins, pole_vel)
    return get_bin(cart_pos_bins, cart_pos), get_bin(cart_vel_bins, cart_vel), get_bin(pole_ang_bins,
                                                                                       pole_ang), get_bin(pole_vel_bins,
                                                                                                          pole_vel)


def q_argmax(d):
    if len(d) == 0:
        return random.randint(0, 1)
    else:
        return max(d.items(), key=itemgetter(1))[0]


def q_max(d):
    if len(d) == 0:
        return 0.0
    else:
        return max(d.items(), key=itemgetter(1))[1]


def ql_update(observation, action, reward, next_observation):
    Q[observation][action] = Q[observation][action] + \
                             learning_rate * (reward + discount_factor * q_max(Q[next_observation]) -
                                              Q[observation][action])


def epsilon_greedy_select(observation, epsilon):
    r = random.random()
    if r < epsilon:
        return random.randint(0, 1)
    else:
        return q_argmax(Q[observation])


def play_one_episode(env, epsilon):
    observation = get_discreet_state(*env.reset())
    done = False
    t = 0
    while not done and t < 10000:
        t += 1
        # Choose E-greedy action
        action = epsilon_greedy_select(observation, epsilon)
        # Take action, observe
        next_observation, reward, done, info = env.step(action)
        next_observation = get_discreet_state(*next_observation)
        # Adjust reward if episode ended
        if done and t < 199:
            reward = -300
        # Update
        ql_update(observation, action, reward, next_observation)
        if done:
            break
        observation = next_observation

    return t


def play_multiple_episodes(env, episodes):
    episode_lengths = np.empty(episodes)

    for i in range(episodes):
        epsilon = 1.0 / np.sqrt(1 + i)
        episode_lengths[i] = play_one_episode(env, epsilon)

    return episode_lengths


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    episode_lengths = play_multiple_episodes(env, 10000)
    plt.plot(episode_lengths)
    plt.show()

    # play a final set of episodes
    # env = wrappers.Monitor(env, 'my_awesome_dir', force=True)
