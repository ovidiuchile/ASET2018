from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt


def load_label_mapping():
    mapping = {}
    with open(r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-mapping.txt', 'rt') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            label, ascii_code = line.split(' ')
            label, ascii_code = int(label), int(ascii_code)
            mapping[label] = chr(ascii_code)
    return mapping


def plot_digit(_mat, val_label):
    _mat = _mat.reshape(28, 28)
    for line in range(len(_mat)):
        for col in range(len(_mat)):
            val = _mat.item(28 * line + col)
            if val > 0.3:
                plt.plot(line, 28 - col, 'ko', alpha=1)

    plt.axis([0, 28, 0, 28])
    plt.title(val_label)
    print(val_label)
    plt.show()


def main():
    train_x, train_y = loadlocal_mnist(
        images_path=r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-train-images-idx3-ubyte',
        labels_path=r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-train-labels-idx1-ubyte')
    test_x, test_y = loadlocal_mnist(
        images_path=r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-test-images-idx3-ubyte',
        labels_path=r'C:\Users\ochile\Desktop\fac\master\RN\tema3\by_merge\emnist-bymerge-test-labels-idx1-ubyte')

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    label_map = load_label_mapping()
    plot_digit(train_x[46], label_map[train_y[46]])


if __name__ == '__main__':
    main()
