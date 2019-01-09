import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn.utils import class_weight 
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import SGD, Adadelta, Adam
from keras import regularizers


def load_label_mapping(mapping_filepath):
    mapping = {}
    with open(mapping_filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            label, ascii_code = line.split(' ')
            label, ascii_code = int(label), int(ascii_code)
            mapping[label] = chr(ascii_code)
    return mapping


def train_8735_8608(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(100, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(decay=1e-6), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=5, batch_size=256)
    model.save('emnist_model.h5')


def train_8655(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(200, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(decay=1e-6), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=5, batch_size=256)
    model.save('emnist_model.h5')


def train_8650(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.01))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(200, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(decay=1e-6), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=5, batch_size=256)
    model.save('emnist_model.h5')


def train_8681(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(100, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(decay=1e-6), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=5, batch_size=256)
    model.save('emnist_model.h5')


def train_8737(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(decay=1e-6), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=5, batch_size=256)
    model.save('emnist_model.h5')


def train_8750(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=5, batch_size=256)
    model.save('emnist_model.h5')


def train_8702(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(600, activation='sigmoid'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=5, batch_size=32)
    model.save('emnist_model.h5')


def train_8720(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(400, activation='sigmoid'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(400, activation='tanh'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=5, batch_size=32)
    model.save('emnist_model.h5')


def train_8832(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=10, batch_size=256)
    model.save('emnist_model.h5')


def train_8999(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256)
    model.save('emnist_model.h5')


def train_8840_la_testare(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=1024)
    model.save('emnist_model.h5')


def train_8897_la_testare(train_x, train_y):
    # 8897 la testare, 9029 la antrenare
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(100, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=1024)
    model.save('emnist_model.h5')


def train_8648_la_testare(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=20, batch_size=128)
    model.save('emnist_model.h5')


def train_8700_la_testare(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.1))
    model.add(Dense(1000, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=20, batch_size=128)
    model.save('emnist_model.h5')


def train_8870_la_testare(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.1))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256)
    model.save('emnist_model.h5')


def train_8845_la_testare(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.15))
    model.add(Dense(1000, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.15))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.15))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.15))
    model.add(Dense(47, activation='softmax'))

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256)
    model.save('emnist_model.h5')


def train_8860_la_testare_params(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.1))
    model.add(Dense(800, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(47, activation='softmax'))

    model.summary()

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256)
    model.save('emnist_model.h5')


def train_8433_la_testare_params(train_x, train_y):
    train_x = train_x / 255
    model = Sequential()
    model.add(InputLayer((784,)))
    # model.add(Dropout(0.1))
    model.add(Dense(800, activation='sigmoid', kernel_initializer='lecun_normal'))
    # model.add(Dropout(0.1))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    # model.add(Dropout(0.1))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    # model.add(Dropout(0.1))
    model.add(Dense(47, activation='softmax'))

    model.summary()

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256)
    model.save('emnist_model.h5')


def train_8675_la_testare_params(train_x, train_y):
    train_x = train_x / 255
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.1))
    model.add(Dense(800, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(47, activation='softmax'))

    model.summary()

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256)
    model.save('emnist_model.h5')


def train_8838_la_testare_params(train_x, train_y):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.15))
    model.add(Dense(800, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.15))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.15))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.15))
    model.add(Dense(47, activation='softmax'))

    model.summary()

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256)
    model.save('emnist_model.h5')


def train_8858_la_testare_params(train_x, train_y, train):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.1))
    model.add(Dense(800, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(47, activation='softmax'))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_y_ints), train_y_ints)

    model.summary()
    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256, class_weight=class_weights)
    model.save('emnist_model.h5')


def train_8860_la_testare_params_v2(train_x, train_y, train_y_ints):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.1))
    model.add(Dense(800, activation='sigmoid', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(400, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='tanh', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(47, activation='softmax'))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_y_ints), train_y_ints)

    model.summary()
    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256, class_weight=class_weights)
    model.save('emnist_model.h5')


def train_8856_la_testare_params_v2(train_x, train_y, train_y_ints):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dropout(0.1))
    model.add(Dense(800, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(400, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='tanh', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.1))
    model.add(Dense(47, activation='softmax'))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_y_ints), train_y_ints)

    model.summary()
    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256, class_weight=class_weights)
    model.save('emnist_model.h5')


def train(train_x, train_y, train_y_ints):
    model = Sequential()
    model.add(InputLayer((784,)))
    model.add(Dense(600, activation='sigmoid', kernel_initializer='lecun_normal'))
    model.add(Dense(400, activation='relu', kernel_initializer='lecun_normal'))
    model.add(Dense(100, activation='tanh', kernel_initializer='lecun_normal'))
    model.add(Dense(47, activation='softmax'))

    model.summary()

    model.compile(optimizer=Adadelta(lr=1.5, decay=1e-5), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(train_x, train_y, epochs=25, batch_size=256)
    model.save('emnist_model.h5')


def main():
    mapping = load_label_mapping(r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-mapping.txt')
    train_x, train_y = loadlocal_mnist(
        images_path=r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-train-images-idx3-ubyte',
        labels_path=r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-train-labels-idx1-ubyte')
    test_x, test_y = loadlocal_mnist(
        images_path=r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-test-images-idx3-ubyte',
        labels_path=r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-test-labels-idx1-ubyte')

    targets = np.eye(47)
    train_y_ints = np.copy(train_y)
    train_y, test_y = [targets[y] for y in [train_y, test_y]]

    print('train_x', train_x.shape)
    print('train_y', train_y.shape)
    train(train_x, train_y, train_y_ints)


if __name__ == '__main__':
    main()
