import numpy
import tensorflow as tf


class SimpleGD:
    def __init__(self, session, loss, model_parameters, parameter, label, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.session = session
        self.loss = loss
        self.w = model_parameters[0]
        self.b = model_parameters[1]
        self.parameter = parameter
        self.label = label
        self.gradients = tf.gradients(loss, [self.w, self.b])

    def partial_fit(self, x, y):
        for model_parameter, gradient in zip([self.w, self.b], self.gradients):
            new_model_parameter = self.session.run((model_parameter - self.learning_rate * gradient),
                                                    feed_dict={self.parameter: x, self.label: y})
            self.session.run(model_parameter.assign(new_model_parameter))


class SimpleSGDRegressor():
    def __init__(self, session, dimensions, learning_rate):
        self.session = session
        self.learning_rate = learning_rate
        self.w = tf.get_variable('w', shape=(1, dimensions), initializer=tf.zeros_initializer(), dtype=tf.float32)
        self.b = tf.get_variable('b', shape=(), initializer=tf.zeros_initializer(), dtype=tf.float32)
        y = tf.get_variable('y', shape=(), initializer=tf.zeros_initializer(), dtype=tf.float32)
        init = tf.global_variables_initializer()
        self.session.run(init)
        model_parameters = [self.w, self.b]
        parameter = tf.placeholder(shape=(1, dimensions), dtype=tf.float32)
        loss = (y - tf.matmul(self.w, tf.transpose(parameter)) - self.b) ** 2
        self.gradient_descent = SimpleGD(self.session, loss, model_parameters, parameter, y, self.learning_rate)

    def partial_fit(self, x, y):
        self.gradient_descent.partial_fit(x, y)

    def fit(self, X, Y):
        for x, y in zip(X, Y):
            self.partial_fit(x.reshape((1, -1)), y)

    def predict(self, x):
        return self.session.run(tf.matmul(self.w, tf.transpose(x)) + self.b).ravel()


class SGDModel:
    def __init__(self, environment, feature_transformer, learning_rate=0.01):
        self.session = tf.Session()
        self.env = environment
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(environment.action_space.n):
            #model = SGDRegressor(learning_rate='constant', max_iter=1000, tol=learning_rate)
            model = SimpleSGDRegressor(self.session, len(environment.action_space.sample()), learning_rate)
            model.partial_fit(feature_transformer.transform([environment.reset()]), [0])
            self.models.append(model)

    def predict(self, state):
        transformed_state = self.feature_transformer.transform([state])
        result = numpy.stack([m.predict(transformed_state) for m in self.models]).T
        assert (len(result.shape) == 2)
        return result

    def update(self, state, action, state_action_value):
        transformed_state = self.feature_transformer.transform([state])
        assert (len(transformed_state.shape) == 2)
        self.models[action].partial_fit(transformed_state, [state_action_value])

    def sample_action(self, state, epsilon):
        if numpy.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return numpy.argmax(self.predict(state))
