import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from decorators import logging_decorator


class NN(object):
    def __init__(self):
        np.set_printoptions(suppress=True)
        self.w1 = np.random.normal(0, 1.0 / math.sqrt(784), (100, 784))
        self.b1 = np.random.normal(0, 1.0 / math.sqrt(100), (100))
        self.w2 = np.random.normal(0, 1.0 / math.sqrt(100), (10, 100))
        self.b2 = np.random.normal(0, 1.0 / math.sqrt(10), (10))

    @staticmethod
    def load_mnist():
        with open('mnist.pkl', 'rb') as fhr:
            train_set, valid_set, test_set = \
                pickle.load(fhr, encoding='latin1')
        return train_set, valid_set, test_set

    @staticmethod
    def get_matrix(_array, dim=28):
        _matrix = [[_array[i * dim + j]
                    for j in range(dim)] for i in range(dim)]
        return np.array(_matrix)

    @staticmethod
    def plot_digit(_mat, val_label, dim=28, is_matrix=False):
        if is_matrix:
            _mat = NN.get_matrix(_mat)
        for line in range(len(_mat)):
            for col in range(len(_mat)):
                val = _mat.item(28 * line + col)
                if val > 0.3:
                    plt.plot(col, dim - line, 'ko', alpha=1)

        plt.axis([0, dim, 0, dim])
        plt.title(val_label)
        plt.show()

    @staticmethod    
    def get_batches(train_set, batch_sz=500):
        batches = []
        train_set_len = len(train_set[0])
        for i in range(int(train_set_len / batch_sz)):
            batches.append(
                [train_set[0][i * batch_sz: i * batch_sz + batch_sz],
                 train_set[1][i * batch_sz: i * batch_sz + batch_sz]])
        return batches

    @staticmethod
    def get_target(t):
        if type(t) in [np.int64, np.int32, np.int16, np.int8, int]:
            return np.array([1 if i == t else 0 for i in range(10)])
        else:
            return np.array([NN.get_target(t[i])
                             for i in range(len(t))])
    
    def sigmoid(self, x):
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sg = self.sigmoid(x)
        return sg * (1 - sg)

    def softmax(self, x):
        x = np.clip(x, -100, 100)
        xexp = np.exp(x)
        xsum = np.sum(xexp)
        return xexp / xsum

    def softmax_4batch(self, x):
        return np.apply_along_axis(self.softmax, 0, x)

    @logging_decorator
    def train(self, train_set, l2=0.1, batch_sz=500,
              eta=0.01, alpha=0.5, epochs=20):
        batches = NN.get_batches(train_set, batch_sz)
        w1_prev = np.zeros(self.w1.shape)
        w2_prev = np.zeros(self.w2.shape)
        for i in tqdm(range(epochs)):
            print(' Epoch %d' % i)
            eta = eta / (1.0 + 0.001 * i)

            for batch in batches:
                z1 = batch[0]

                z2 = self.w1.dot(z1.T) + \
                     np.repeat(self.b1, batch_sz).reshape(100, batch_sz)
                y2 = self.sigmoid(z2)

                z3 = self.w2.dot(y2) + \
                     np.repeat(self.b2, batch_sz).reshape(10, batch_sz)
                y3 = self.softmax_4batch(z3)

                target = NN.get_target(batch[1]).T

                err3 = y3 - target
                new_w2 = (err3.dot(y2.T) + (self.w2 * l2)) * eta

                err2 = np.multiply(self.w2.T.dot(err3),
                                   np.multiply(y2, 1 - y2))
                new_w1 = (err2.dot(z1) + (self.w1 * l2)) * eta

                self.w2 -= ((1 - alpha) * new_w2 + (alpha * w2_prev))
                self.b2 -= eta * np.sum(new_w2.T, axis=0) / batch_sz
                self.w1 -= ((1 - alpha) * new_w1 + (alpha * w1_prev))
                self.b1 -= eta * np.sum(new_w1.T, axis=0) / batch_sz

                w1_prev, w2_prev = new_w1, new_w2

            if test_set:
                self.test(test_set)
    @logging_decorator
    def predict(self, X):
        z2 = self.w1.dot(X[0].T)
        y2 = self.sigmoid(z2)

        z3 = self.w2.dot(y2)
        y3 = self.softmax_4batch(z3)

        return np.argmax(y3, axis=0)

    @logging_decorator
    def test(self, test_set):
        argmax = net.predict(test_set)
        acc = float(np.sum(test_set[1] == argmax)) / test_set[0].shape[0]
        print('Testing accuracy: %.2f%%' % (acc * 100))
        return '%.4f' % (acc * 100)


train_set, valid_set, test_set = NN.load_mnist()
train_set_vals = np.concatenate((train_set[0], valid_set[0]), axis=0)
train_set_labs = np.concatenate((train_set[1], valid_set[1]), axis=0)

train_set = (train_set_vals, train_set_labs)
net = NN()

net.train(train_set)

acc = net.test(test_set)
with open('acc_%s.pkl' % acc, 'wb') as fhw:
    fhw.write(pickle.dumps(net))
