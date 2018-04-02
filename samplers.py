import numpy.random
import numpy


class CompoundSampler:
    def __init__(self, environment, actions):
        self.state_sampler = environment.observation_space
        self.actions = actions

    def sample(self):
        state_sample = self.state_sampler.sample()
        action_sample = numpy.random.sample(self.actions)
        return numpy.hstack(state_sample, action_sample)
