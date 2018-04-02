import numpy
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


class RBFFeatureTransformer:
    def __init__(self, sampler, gammas, n_components=500, n_samples=20000):
        observation_examples = numpy.array([sampler.sample() for _ in range(n_samples)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        self.gammas = gammas

        samplers = [("rbf{}".format(i), RBFSampler(gamma=gammas[i], n_components=n_components)) for i in range(4)]

        featurizer = FeatureUnion(samplers)
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


class CartPoleRBFFeatureTransformer(RBFFeatureTransformer):
    def __init__(self, environment):
        gammas = [0.05, 1.0, 0.5, 0.1]
        super(CartPoleRBFFeatureTransformer, self).__init__(environment, gammas, n_components=1000)


class MountainCarRBFFeatureTransformer(RBFFeatureTransformer):
    def __init__(self, environment):
        gammas = [5.0, 2.0, 1.0, 0.5]
        super(MountainCarRBFFeatureTransformer, self).__init__(environment, gammas)


class MountainCarCompoundRBFFeatureTransformer:
    def __init__(self, environment):
        self.state_feature_transformer = MountainCarRBFFeatureTransformer(environment)

    def transform(self, observations, actions):
        transformed_state = self.state_feature_transformer(observations)
        return numpy.hstack(transformed_state, actions)
