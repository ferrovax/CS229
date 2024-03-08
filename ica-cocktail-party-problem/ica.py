import numpy as np
import scipy.io.wavfile
import os
from numpy import linalg as LA

def update_W(W, x, learning_rate):
    """
    Performs gradient ascent.
    Args:
        W: W matrix for ICA
        x: a data element
        learning_rate: learning rate
    Returns:
        updated W
    """
    updated_W = W + learning_rate * (-np.outer(np.sign(W.dot(x)), x) + LA.inv(W.T))
    return updated_W


def unmix(X, W):
    """
    Unmix an X matrix according to W using ICA.
    Args:
        X: data matrix
        W: W for ICA
    Returns:
        numpy array S containing the split data
    """

    S = np.zeros(X.shape)
    S = X.dot(W.T)
    
    return S


Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('./mix.dat')
    return mix

def save_W(W):
    np.savetxt('./W.txt',W)

def save_sound(audio, name):
    scipy.io.wavfile.write('./{}.wav'.format(name), Fs, audio)

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1 , 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01 , 0.01, 0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    for lr in anneal:
        print(lr)
        rand = np.random.permutation(range(M))
        for i in rand:
            x = X[i]
            W = update_W(W, x, lr)

    return W

def main():
    np.random.seed(0) # random seed for reproducibility
    X = normalize(load_data())
    print(X.shape)

    for i in range(X.shape[1]):
        save_sound(X[:, i], 'mixed_{}'.format(i))

    W = unmixer(X)
    print(W)
    save_W(W)
    
    S = normalize(unmix(X, W))
    assert S.shape[1] == 5
    
    for i in range(S.shape[1]):
        if os.path.exists('split_{}'.format(i)):
            os.unlink('split_{}'.format(i))
        save_sound(S[:, i], 'split_{}'.format(i))

if __name__ == '__main__':
    main()
