import numpy


class SimpleSGDRegressor():
    def __init__(self, dimensions, learning_rate):
        self.w = numpy.zeros(shape=(1, dimensions))
        self.b = numpy.zeros(shape=(1,))
        self.learning_rate = learning_rate

    def gradients(self, x, y):
        dl_dw = -(y - self.predict(x))
        return numpy.dot(dl_dw, x), dl_dw

    def partial_fit(self, x, y):
        dw, db = self.gradients(x, y)
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def fit(self, X, Y):
        for x, y in zip(X, Y):
            self.partial_fit(x.reshape((1, -1)), y)

    def predict(self, x):
        return (numpy.dot(self.w, x.T) + self.b).ravel()


class SGDStateActionModel:
    def __init__(self, environment, feature_transformer, learning_rate=0.01):
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(environment.action_space.n):
            model = SimpleSGDRegressor(feature_transformer.dimensions, learning_rate)
            model.partial_fit(feature_transformer.transform([environment.reset()]), [0])
            self.models.append(model)

    def predict(self, state):
        transformed_state = self.feature_transformer.transform([state])
        result = numpy.stack([m.predict(transformed_state.astype(numpy.float32)) for m in self.models]).T
        return result

    def update(self, state, action, state_action_value):
        transformed_state = self.feature_transformer.transform([state])
        self.models[action].partial_fit(transformed_state.astype(numpy.float32), [state_action_value])

    def sample_action(self, state, epsilon):
        if numpy.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return numpy.argmax(self.predict(state))


class SGDStateModel:
    def __init__(self, environment, feature_transformer, learning_rate=0.01):
        self.regressor = SimpleSGDRegressor(feature_transformer.dimensions, learning_rate)
        self.regressor.partial_fit(feature_transformer.transform([environment.reset()]), [0])
        self.feature_transformer = feature_transformer

    def predict(self, state):
        transformed_state = self.feature_transformer.transform([state])
        return self.regressor.predict(transformed_state.astype(numpy.float32))

    def update(self, state, state_value):
        transformed_state = self.feature_transformer.transform([state])
        self.regressor.partial_fit(transformed_state.astype(numpy.float32), [state_value])


class ActorCriticModel:
    def __init__(self, environment, feature_transformer):
        pass

    def update(self, state, action, target):
        pass

    def sample_action(self, state):
        pass
