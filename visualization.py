import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import numpy


def plot_running_avg(total_rewards, output_path='running_average.png', window_size=100):
    size = len(total_rewards)
    running_avg = numpy.empty(size)
    for t in range(size):
        running_avg[t] = total_rewards[max(0, t - window_size):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.savefig(output_path)


def plot_cost_to_go(observation_space, estimator, output_path='cost_to_go.png',
                    labels=['Position', 'Velocity'], num_tiles=20):
    x = numpy.linspace(observation_space.low[0], observation_space.high[0], num=num_tiles)
    y = numpy.linspace(observation_space.low[1], observation_space.high[1], num=num_tiles)
    X, Y = numpy.meshgrid(x, y)
    Z = numpy.apply_along_axis(lambda _: -numpy.max(estimator.predict(_)), 2, numpy.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
        rstride=1, cstride=1, cmap=colormap.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.savefig(output_path)
