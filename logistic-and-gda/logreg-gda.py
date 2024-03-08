import numpy as np
import util

WILDCARD = 'X'

def main_LogReg(train_path, valid_path, save_path):
    """
    Newton's method (logistic regression)
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)

    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('LR Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

class LogisticRegression:
    """
    usage:
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        args:
            step_size: for iterative solvers
            max_iter: max iterations
            eps: convergence threshold
            theta_0: init theta
            verbose: prints loss if True
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.sigmoid = lambda _ : 1 / ( 1 + np.exp(-1 * _) )

    def grad(self, x, y):
        return 1 / x.shape[0] * x.T.dot(self.sigmoid(x.dot(self.theta)) - y)

    def hess(self, x):
        _ = self.sigmoid(x.dot(self.theta))
        diag = np.diag(_ * (1 - _))
        return 1 / x.shape[0] * x.T.dot(diag).dot(x)

    def loss(self, x, y):
        h = self.sigmoid(x.dot(self.theta))
        return -1 * np.mean(y * np.log(h + self.eps) + (1 - y) * np.log(1 - h + self.eps))

    def fit(self, x, y):
        """
        minimize J(theta) with Newton's method
        args:
            x: inputs, shape (n_examples, dim)
            y: labels, shape (n_examples,)
        """
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for iter in range(self.max_iter):
            gradient = self.grad(x, y)
            H = self.hess(x)

            theta = np.copy(self.theta)
            self.theta -= self.step_size * np.linalg.inv(H).dot(gradient)

            if self.verbose:
                print(f'Iteration {iter}, Loss {self.loss(x,y)}')

            if self.eps > np.sum(np.abs(theta - self.theta)):
                break

    def predict(self, x):
        """
        args:
            x: inputs (n_examples, dim)
        returns:
            outputs (n_examples,)
        """
        return self.sigmoid(x.dot(self.theta))

def main_GDA(train_path, valid_path, save_path):
    """
    Gaussian discriminant analysis (GDA)
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    
    clf = GDA()
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)
    x_eval = util.add_intercept(x_eval)

    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('GDA Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

class GDA:
    """
    usage:
        clf = GDA()
        clf.fit(x_train, y_train)
        clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        args:
            step_size: for iterative solvers
            max_iter: max iterations
            eps: convergence threshold
            theta_0: init theta
            verbose: prints loss if True
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.sigmoid = lambda _ : 1 / ( 1 + np.exp(-1 * _) )

    def fit(self, x, y):
        """
        args:
            x: inputs (n_examples, dim)
            y: labels (n_examples,)
        """
        phi = 1 / x.shape[0] * np.sum(y == 1)
        mu0 = (y == 0).dot(x) / np.sum(y == 0)
        mu1 = (y == 1).dot(x) / np.sum(y == 1)

        muy = np.where(np.expand_dims(y == 0, -1), np.expand_dims(mu0, 0), np.expand_dims(mu1, 0))
        sig = 1 / x.shape[0] * (x - muy).T.dot(x - muy)

        self.theta = np.zeros(1 + x.shape[1])
        inv = np.linalg.inv(sig)

        self.theta[0] = 1 / 2 * (mu0.T.dot(inv).dot(mu0) - mu1.T.dot(inv).dot(mu1)) - np.log((1 - phi) / phi)
        self.theta[1:] = -inv.dot(mu0 - mu1)

    def predict(self, x):
        """
        args:
            x: inputs (n_examples, dim)
        returns:
            outputs (n_examples,)
        """
        return self.sigmoid(x.dot(self.theta))

def main_posonly(train_path, valid_path, test_path, save_path):
    """
    logistic regression for incomplete, positive-only labels
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    plot_path = save_path.replace('.txt', '.png')
    plot_path_true = plot_path.replace(WILDCARD, 'true')
    plot_path_naive = plot_path.replace(WILDCARD, 'naive')
    plot_path_adjusted = plot_path.replace(WILDCARD, 'adjusted')

    # train and test on true labels (t)
    full_predictions = fully_observed_predictions(train_path, test_path, output_path_true, plot_path_true)

    # train on y-labels and test on true labels
    naive_predictions, clf = naive_partial_labels_predictions(train_path, test_path, output_path_naive, plot_path_naive)

    # apply correction factor using validation set and test on true labels
    alpha = find_alpha_and_plot_correction(clf, valid_path, test_path, output_path_adjusted, plot_path_adjusted, naive_predictions)

    return

def fully_observed_predictions(train_path, test_path, output_path_true, plot_path_true):
    """
    helper fn for fully observable binary classification
    
    return:
        full_predictions: prediction tensor
    """
    full_predictions = None
    # train and test on true labels (t)
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, t_train)

    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    full_predictions = clf.predict(x_test)

    np.savetxt(output_path_true, full_predictions)

    return full_predictions

def naive_partial_labels_predictions(train_path, test_path, output_path_naive, plot_path_naive):
    """
    helper fn for naive partial labels binary classification

    return:
        naive_predictions: prediction tensor
        clf: logistic regression classifier
    """
    naive_predictions = None
    clf = None
    # train on y-labels, test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    naive_predictions = clf.predict(x_test)

    np.savetxt(output_path_naive, naive_predictions)

    return naive_predictions, clf

def find_alpha_and_plot_correction(clf, valid_path, test_path, output_path_adjusted, plot_path_adjusted, naive_predictions):
    """
    helper fn for alpha correction binary classification
    
    return:
        alpha: corrected alpha value
    """
    alpha = None
    
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y')

    x_valid = x_valid[y_valid == 1, :]
    x_valid = util.add_intercept(x_valid)

    y_pred = clf.predict(x_valid)
    alpha = np.mean(y_pred)

    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    np.savetxt(output_path_adjusted, naive_predictions / alpha)

    return alpha

if __name__ == '__main__':
    # linear classifiers
    main_LogReg(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')
    main_LogReg(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
    main_GDA(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')
    main_GDA(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')

    # incomplete, positive-only labels
    main_posonly(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
