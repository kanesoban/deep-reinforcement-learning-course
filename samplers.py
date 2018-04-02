import random
import numpy


class CompoundSampler:
    def __init__(self, environment, actions):
        self.state_sampler = environment.observation_space
        self.actions = actions

    def sample(self):
        state_sample = self.state_sampler.sample()
        action_sample = numpy.array(random.sample([0, 1], 1))
        return numpy.hstack((state_sample, action_sample))
