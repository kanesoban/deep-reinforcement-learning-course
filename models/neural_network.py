import tensorflow as tf
import numpy


class NeuralNetwork:
    def __init__(self, session, input, layers, learning_rate, scope='model'):
        self.session = session
        self.input = input
        self.layers = layers
        with tf.variable_scope(scope):
            self.target = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.loss_function = tf.reduce_mean(tf.square(self.target - self.layers[-1]))
            self.learning_rate = learning_rate
            self.predict_op = self.layers[-1]
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_function)

    def predict(self, input):
        return self.session.run(
            self.predict_op,
            feed_dict={
                self.input: input
            }
        )

    def update(self, input, target):
        return self.session.run(
            self.train_op,
            feed_dict={
                self.input: input,
                self.target: target
            }
        )


class MountainCarNeuralNetwork(NeuralNetwork):
    def __init__(self, session, input_size, num_actions, environment, learning_rate=0.001, scope='mountaincar_network'):
        self.environment = environment
        with tf.variable_scope(scope):
            input = tf.placeholder(tf.float32, shape=(input_size,), name='X')
            layers = []
            output = tf.contrib.layers.fully_connected(input, num_actions)
            layers.append(output)
            output = tf.contrib.layers.fully_connected(output, num_actions)
            layers.append(output)
            super(NeuralNetwork).__init__(self, session, input, layers, learning_rate)

    def sample_action(self, state, epsilon):
        if numpy.random.random() < epsilon:
            return self.environment.action_space.sample()
        else:
            return numpy.argmax(self.predict(state))
