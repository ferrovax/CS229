import math

import matplotlib.pyplot as plt
import numpy as np

import util


def initial_state():
    return []


def predict(state, kernel, x_i):
    """
    args:
        state: state
        kernel: binary function, takes two vectors as input
        x_i: feature vector
    returns:
        prediction (0 or 1)
    """
    _ = sum([kernel(x, x_i) * w for w, x in state])
    return sign(_)


def update_state(state, kernel, learning_rate, x_i, y_i):
    """
    args:
        state: state
        kernel: binary function, takes two vectors as input
        learning_rate: learning rate
        x_i: feature vector
        y_i: label
    """
    state.append( (learning_rate * (y_i - predict(state, kernel, x_i)), x_i) )


def sign(a):
    """
    sign getter
    """
    if a >= 0:
        return 1
    else:
        return 0


def dot_kernel(a, b):
    """
    a dot product kernel

    args:
        a: vector a
        b: vector b
    """
    return np.dot(a, b)


def rbf_kernel(a, b, sigma=1):
    """
    radial basis function kernel

    args:
        a: vector a
        b: vector b
        sigma: kernel radius
    """
    distance = (a - b).dot(a - b)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return math.exp(scaled_distance)


def train_perceptron(kernel_name, kernel, learning_rate):
    train_x, train_y = util.load_csv('train.csv')

    state = initial_state()

    for x_i, y_i in zip(train_x, train_y):
        update_state(state, kernel, learning_rate, x_i, y_i)

    test_x, test_y = util.load_csv('test.csv')

    plt.figure(figsize=(12, 8))
    util.plot_contour(lambda a: predict(state, kernel, a))
    util.plot_points(test_x, test_y)
    plt.savefig('perceptron_{}_output.png'.format(kernel_name))

    predict_y = [predict(state, kernel, test_x[i, :]) for i in range(test_y.shape[0])]

    np.savetxt('perceptron_{}_predictions'.format(kernel_name), predict_y)


def main():
    train_perceptron('dot', dot_kernel, 0.5)
    train_perceptron('rbf', rbf_kernel, 0.5)


if __name__ == "__main__":
    main()
