import numpy as np
from scipy.io import loadmat
from keras.utils import np_utils
from keras.models import load_model
import random
import matplotlib.pyplot as plt

width = 28
height = 28
nb_classes = 26

model = load_model('model.h5')
model.summary()

DATASET = 'emnist-letters.mat'
mat = loadmat(DATASET)

max2 = len(mat['dataset'][0][0][1][0][0][0])  # 20.800
testing_images = mat['dataset'][0][0][1][0][0][0][:max2].reshape(max2, height, width, 1)
testing_labels = np.array([i[0] - 1 for i in mat['dataset'][0][0][1][0][0][1][:max2]])
print('items test', max2)

testing_images = testing_images.astype('float32')
testing_images /= 255


def get_matrix(_array, dim=28):
    _matrix = [[_array[i * dim + j]
                for j in range(dim)] for i in range(dim)]
    return np.array(_matrix)


def plot_digit(_mat, val_label):
    _mat = _mat.reshape(28, 28)
    for line in range(len(_mat)):
        for col in range(len(_mat)):
            val = _mat.item(28 * line + col)
            if val > 0.3:
                plt.plot(line, 28 - col, 'ko', alpha=1)

    plt.axis([0, 28, 0, 28])
    plt.title(val_label)
    plt.show()


x_test_all, y_test = testing_images, np_utils.to_categorical(testing_labels, nb_classes)


def get_prediction_with_neural_network(image):
    output = model.predict(image)
    return chr(np.argmax(output) + 97)


def predict(i):
    try:
        image = x_test_all[i].reshape(1, 28, 28, 1)
        letter = get_prediction_with_neural_network(image)
        print('Predicted ok:', letter)
        return image, letter
    except BaseException:
        print("Invalid Image")
        return None, None


for _ in range(10):
    i = random.randint(0, len(x_test_all))
    image, letter = predict(i)
    plot_digit(image, letter)
