import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """
    poisson regression with gradient ascent
    args:
        lr: learning rate
        train_path: path to csv
        eval_path: path to csv
        save_path: save path
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    p_eval = clf.predict(x_eval)
    np.savetxt(save_path, p_eval)
    plt.figure()
    plt.scatter(y_eval,p_eval,alpha=0.4,c='red',label='Ground Truth vs Predicted')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.legend()
    plt.savefig('poisson_valid.png')

class PoissonRegression:
    """
    usage:
        clf = PoissonRegression(step_size=lr)
        clf.fit(x_train, y_train)
        clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        args:
            step_size: step size for iterative solvers
            max_iter: max iterations
            eps: convergence threshold
            theta_0: init theta
            verbose: whether to print loss
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """
        args:
            x: inputs (n_examples, dim)
            y: labels (n_examples,)
        """
        i, previous = 0, None

        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        while previous is None or np.linalg.norm(self.theta) - np.linalg.norm(previous) > self.eps and self.max_iter > i:
            previous = np.copy(self.theta)
            gradient = np.expand_dims(y - self.predict(x), 1) * x
            self.theta += self.step_size * np.sum(gradient, axis=0)

            i += 1

    def predict(self, x):
        """
        args:
            x: inputs (n_examples, dim)
        returns:
            predictions (n_examples,)
        """
        return np.exp(x.dot(self.theta))

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
