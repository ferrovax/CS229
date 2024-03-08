import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
  def __init__(self, theta=None):
    self.theta = theta

  def fit(self, x, y):
    self.theta = np.linalg.solve(x.T.dot(x), x.T.dot(y))

  def predict(self, x):
    return x.dot(self.theta)

  @staticmethod
  def create_poly(k, x):
    """
    args:
      x: training input
    returns: 
      polynomial features of x with powers 0 to k (inclusive)
    """
    return x**np.arange(k+1)

  @staticmethod
  def create_sin(k, x):
    """
    generate sine and polynomial features of x
    args:
      x: training input
    returns: sine (column 0) and polynomial (columns 1 to k+1) features of x with powers 0 to k (inclusive)
    """
    return np.c_[np.sin(x), x**np.arange(k+1)]

def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
  train_x,train_y=util.load_dataset(train_path,add_intercept=False)
  plot_x = np.ones([1000, 1])
  plot_x[:, 0] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
  plt.figure()
  plt.scatter(train_x, train_y)

  for k in ks:
      if sine:
          model = LinearModel()
          model.fit(model.create_sin(k, train_x), train_y)
          feat = model.create_sin(k, plot_x)
          plot_y = model.predict(feat)
      else:
          model = LinearModel()
          model.fit(model.create_poly(k, train_x), train_y)
          feat = model.create_poly(k, plot_x)
          plot_y = model.predict(feat)
        
      plt.ylim(-2, 2)
      plt.plot(plot_x[:, 0], plot_y, label='k=%d' % k)

  plt.legend()
  plt.savefig(filename)
  plt.clf()


def main(train_path, small_path, eval_path):
  run_exp(train_path, True, [1, 2, 3, 5, 10, 20], 'large-sine.png')
  run_exp(train_path, False, [1, 2, 3, 5, 10, 20], 'large-poly.png')
  run_exp(small_path, True, [1, 2, 3, 5, 10, 20], 'small-sine.png')
  run_exp(small_path, False, [1, 2, 3, 5, 10, 20], 'small-poly.png')

if __name__ == '__main__':
  main(train_path='train.csv',
      small_path='small.csv',
      eval_path='test.csv')
