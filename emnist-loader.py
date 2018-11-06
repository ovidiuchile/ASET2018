import pickle
import keras
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat
import numpy as np

DATASET = 'emnist-letters.mat'
width = 28
height = 28

mat = loadmat(DATASET)
# mapping = {kv[0]: kv[1:][0] for kv in mat['dataset'][0][0][2]}
nb_classes = 26

max1 = len(mat['dataset'][0][0][0][0][0][0])  # 124.800
training_images = mat['dataset'][0][0][0][0][0][0][:max1].reshape(max1, height, width, 1)
training_labels = np.array([i[0] - 1 for i in mat['dataset'][0][0][0][0][0][1][:max1]])
print('items train', max1)

max2 = len(mat['dataset'][0][0][1][0][0][0])  # 20.800
testing_images = mat['dataset'][0][0][1][0][0][0][:max2].reshape(max2, height, width, 1)
testing_labels = np.array([i[0] - 1 for i in mat['dataset'][0][0][1][0][0][1][:max2]])
print('items test', max2)

input_shape = (28, 28, 1)

# Hyperparameters
nb_filters = 32  # number of convolutional filters to use
pool_size = (2, 2)  # size of pooling area for max pooling
kernel_size = (3, 3)  # convolution kernel size
batch_size = 256
epochs = 10

model = Sequential()
model.add(Convolution2D(nb_filters,
                        kernel_size,
                        padding='valid',
                        input_shape=input_shape,
                        activation='relu'))
model.add(Convolution2D(nb_filters,
                        kernel_size,
                        activation='relu'))

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

training_images = training_images.astype('float32')
testing_images = testing_images.astype('float32')

training_images /= 255
testing_images /= 255

x_train, y_train = training_images, np_utils.to_categorical(training_labels, nb_classes)
x_test, y_test = testing_images, np_utils.to_categorical(testing_labels, nb_classes)


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

save_model(model, 'model.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

