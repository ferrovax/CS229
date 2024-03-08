import numpy as np
import matplotlib.pyplot as plt
import argparse

HPARAMS = {
    'batch_size' : 1000,
    'num_epochs' : 30,
    'learning_rate' : 0.4,
    'num_hidden' : 300,
    'reg' : 0.001
}

def softmax(x):
    _ = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
    return _ / np.sum(_, axis=1)[:, np.newaxis]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_initial_params(input_size, num_hidden, num_output):
    """
    args:
        input_size: input data size
        num_hidden: number of hidden states
        num_output: number of output classes
    returns:
        dict mapping parameter names to numpy arrays
    """
    _ = {
            'W1': np.random.normal(size=(input_size,num_hidden)),
            'W2': np.random.normal(size=(num_hidden,num_output)),
            'b1': np.zeros(num_hidden),
            'b2': np.zeros(num_output)
    }
    return _

def forward_prop(data, labels, params):
    """
    args:
        data: input numpy array
        labels: 2d numpy array of labels
        params: dictionary mapping parameter names to numpy parameter arrays
    returns:
        3 element tuple containing:
            1. numpy array of the activations (after sigmoid) of the hidden layer
            2. numpy array the output (after softmax) of the output layer
            3. average loss
    """
    activations = sigmoid(data.dot(params['W1']) + params['b1'])
    output = softmax(_.dot(params['W2']) + params['b2'])
    loss = np.sum(-labels * np.log(__)) / data.shape[0]

    return activations, output, loss

def backward_prop(data, labels, params, forward_prop_func):
    """
    args:
        data: input numpy array
        labels: 2d numpy array of labels
        params: dictionary mapping parameter names to numpy parameter arrays
        forward_prop_func: function that follows the forward_prop API^^
    returns:
        dictionary of W1, W2, b1, and b2 strings (names) to numpy arrays (gradient)
    """
    return backward_prop_regularized(data, labels, params, forward_prop_func, 0)


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    args:
        data: input numpy array
        labels: 2d numpy array of labels
        params: parameter dictioinary
        forward_prop_func: function that follows the forward_prop API^^
        reg: lambda regularization factor
    returns:
        dictionary of W1, W2, b1, and b2 strings (names) to numpy arrays (gradient)
    """
    h, y, c = forward_prop_func(data, labels, params)

    _w2 = h.T.dot(y - labels) / data.shape[0] + reg * 2 * params['W2']
    _b2 = np.sum(y - labels, axis=0) / data.shape[0]
    _w1 = data.T.dot((y - labels).dot(params['W2'].T) * h * (1 - h)) / data.shape[0] + reg * 2 * params['W1']
    _b1 = np.sum((y - labels).dot(params['W2'].T) * h * (1 - h), axis=0) / data.shape[0]

    return {
            'W1': _w1,
            'W2': _w2,
            'b1': _b1,
            'b2': _b2
    }

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    args:
        train_data: numpy array
        train_labels: numpy array
        learning_rate: learning rate
        batch_size: number items to process per batch
        params: parameter dictionary
        forward_prop_func: function that follows the forward_prop API
        backward_prop_func: function that follows the backwards_prop API
    """
    for i in range(train_data.shape[0] // batch_size):
        _ = backward_prop_func(train_data[i * 1000 : (i+1) * 1000,:],
                                train_labels[i * 1000 : (i+1) * 1000,:],
                                params,
                                forward_prop_func)
        params['W1'] = params['W1'] - learning_rate * _['W1']
        params['W2'] = params['W2'] - learning_rate * _['W2']
        params['b1'] = params['b1'] - learning_rate * _['b1']
        params['b2'] = params['b2'] - learning_rate * _['b2']

    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels,
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=HPARAMS['num_hidden'], learning_rate=HPARAMS['learning_rate'],
    num_epochs=HPARAMS['num_epochs'], batch_size=HPARAMS['batch_size']):

    print(f'Num hidden:    {num_hidden}')
    print(f'Learning rate: {learning_rate}')
    print(f'Num epochs:    {num_epochs}')
    print(f'Bach size:     {batch_size}')

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels,
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) ==
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file, max_rows=None):
    if max_rows is None:
        x = np.loadtxt(images_file, delimiter=',')
        y = np.loadtxt(labels_file, delimiter=',')
    else:
        x = np.loadtxt(images_file, delimiter=',', max_rows = max_rows)
        y = np.loadtxt(labels_file, delimiter=',', max_rows = max_rows)
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True, test_set = False):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'],
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=HPARAMS['num_hidden'], learning_rate=HPARAMS['learning_rate'],
        num_epochs=HPARAMS['num_epochs'], batch_size=HPARAMS['batch_size']
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    if test_set:
        accuracy = nn_test(all_data['test'], all_labels['test'], params)
        print('For model %s, achieved test set accuracy: %f' % (name, accuracy))

def main(num_epochs=HPARAMS['num_epochs'], plot=True, train_baseline = True, train_regularized=True, test_set = False):
    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }

    if train_baseline:
        run_train_test('baseline', all_data, all_labels, backward_prop, num_epochs, plot, test_set = test_set)
    if train_regularized:
        print('Regularization param: ', HPARAMS['reg'])
        run_train_test('regularized', all_data, all_labels,
            lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=HPARAMS['reg']),
            num_epochs, plot, test_set = test_set)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=HPARAMS['num_epochs'])

    args = parser.parse_args()

    main(num_epochs = args.num_epochs)
