# Otsu in Python
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_image(file_name):
    img = Image.open(file_name)
    img.load()
    plt.imshow(img)
    plt.show()
    bw = img.convert('L')
    bw_data = np.array(bw).astype('int16')

    BINS = np.array(range(0, 257))
    counts, pixels = np.histogram(bw_data, BINS)
    pixels = pixels[:-1]
    # plt.bar(pixels, counts, align='center')
    # plt.savefig('histogram.png')
    # plt.xlim(-1, 256)
    # plt.show()

    total_counts = np.sum(counts)
    assert total_counts == bw_data.shape[0] * bw_data.shape[1]

    return BINS, counts, pixels, bw_data, total_counts


def within_class_variance():
    ''' Here we will implement the algorithm and find the lowest Within-Class Variance:
    Refer to this page for more details http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html'''

    for i in range(1, len(BINS), 1):  # from one to 257 = 256 iterations
        prob_1 = np.sum(counts[:i]) / total_counts
        prob_2 = np.sum(counts[i:]) / total_counts
        assert (np.sum(prob_1 + prob_2)) == 1.0

        mean_1 = np.sum(counts[:i] * pixels[:i]) / np.sum(counts[:i])
        mean_2 = np.sum(counts[i:] * pixels[i:]) / np.sum(counts[i:])
        var_1 = np.sum(((pixels[:i] - mean_1)**2) * counts[:i]) / np.sum(counts[:i])
        var_2 = np.sum(((pixels[i:] - mean_2)**2) * counts[i:]) / np.sum(counts[i:])

        if i == 1:
            cost = (prob_1 * var_1) + (prob_2 * var_2)
            keys = {'cost': cost, 'mean_1': mean_1, 'mean_2': mean_2, 'var_1': var_1, 'var_2': var_2, 'pixel': i - 1}
            print('first_cost', cost)

        if (prob_1 * var_1) + (prob_2 * var_2) < cost:
            cost = (prob_1 * var_1) + (prob_2 * var_2)
            keys = {'cost': cost, 'mean_1': mean_1, 'mean_2': mean_2, 'var_1': var_1,
                    'var_2': var_2, 'pixel': i - 1}  # pixels is i-1 because BINS is starting from one

    return keys


def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0 / pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]:  # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = Wb * Wf * (mub - muf) ** 2

        # print("Wb", Wb, "Wf", Wf)
        # print("t", t, "value", value)

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    # print(final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img


if __name__ == "__main__":
    file_name = sys.argv[1]
    BINS, counts, pixels, bw_data, total_counts = load_image(file_name)
    # keys = within_class_variance()
    # print(keys['pixel'])
    # otsu_img = np.copy(bw_data).astype('uint8')
    # otsu_img[otsu_img > keys['pixel']] = 1
    # otsu_img[otsu_img < keys['pixel']] = 0
    # # print(otsu_img.dtype)
    # plt.imshow(otsu_img)
    # plt.savefig('otsu.png')
    # plt.show()

    otsu_img_new = otsu(bw_data)
    plt.imshow(otsu_img_new, cmap=cm.gray)
    plt.savefig('otsu_img_new.png')
    print(otsu_img_new)
    plt.show()
