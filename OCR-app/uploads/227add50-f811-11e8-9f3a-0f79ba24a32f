import sys
import cv2
import numpy as np
from keras.models import load_model


def contours_from_binary(binary_image):
    _, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    result_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), np.uint8)
    # print(hierarchy)

    # draw contours and hull points
    external_hulls = []
    for i in range(len(contours)):
        if hierarchy[0, i, 3] == 0:
            # color_contours = (0, 255, 0)  # green - color for contours
            color = (255, 0, 0)  # blue - color for convex hull
            # draw ith contour
            # cv2.drawContours(result_image, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours(result_image, hull, i, color, 1, 8)
            external_hulls.append(np.reshape(hull[i], (hull[i].shape[0], hull[i].shape[2])))

    for i in range(result_image.shape[0]):
        for j in range(result_image.shape[1]):
            if result_image[i, j, 0] == 0 and result_image[i, j, 1] == 0 and result_image[i, j, 2] == 0:
                result_image[i, j] = (255, 255, 255)

    cv2.imshow('img-windows', result_image)
    cv2.waitKey(0)
    cv2.imwrite('01.png', result_image)
    return external_hulls


def get_letter_images(gray_image, contours):
    letter_images = []

    for contour in contours:
        if isinstance(contour, str) and contour == ' ':
            letter_images.append(' ')
            continue
        min_x = np.min(contour[:, 0])
        min_y = np.min(contour[:, 1])
        max_x = np.max(contour[:, 0])
        max_y = np.max(contour[:, 1])

        min_x = max([min_x - int((max_x - min_x) / 10), 0])
        max_x = min([max_x + int((max_x - min_x) / 10), gray_image.shape[1]])
        min_y = max([min_y - int((max_y - min_y) / 10), 0])
        max_y = min([max_y + int((max_y - min_y) / 10), gray_image.shape[0]])

        letter_images.append(cv2.resize(
            gray_image[min_y:max_y, min_x:max_x],
            (28, 28)
        ))
    return letter_images


def load_rn_model():
    return load_model('../neural_network/emnist_model.h5')


def load_label_mapping():
    mapping_filepath = '../neural_network/by_merge/emnist-bymerge-mapping.txt'
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


def predict(images, model, mapping):
    text = ''
    for image in images:
        if isinstance(image, str) and image == ' ':
            text += ' '
            continue
        image = 255 - image
        image = image.T
        image = np.reshape(image, (1, 784))
        text += mapping[np.argmax(model.predict(image))]
    return text.lower()


def main():
    image = cv2.imread(sys.argv[1], 1)  # read input image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blured_gray_image = cv2.blur(gray_image, (3, 3))  # blur the image

    binary_image = cv2.adaptiveThreshold(blured_gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)

    contours = contours_from_binary(binary_image)
    contours = sorted(contours, key=lambda e: np.mean(e[:, 0]))

    space_mean = 0
    for i in range(len(contours)):
        if i != 0:
            space_mean += np.mean(contours[i]) - np.mean(contours[i - 1])
    space_mean /= len(contours) - 1
    contours_and_spaces = []
    for i in range(len(contours)):
        if i != 0:
            if np.mean(contours[i]) - np.mean(contours[i - 1]) > space_mean * 1.2:
                contours_and_spaces.append(' ')
        contours_and_spaces.append(contours[i])

    letter_images = get_letter_images(gray_image, contours_and_spaces)

    model = load_rn_model()
    mapping = load_label_mapping()
    text = predict(letter_images, model, mapping)
    print(text)

    # print(image.shape)

    # print(contours[0])
    # result_image = np.ones((binary_image.shape[0], binary_image.shape[1], 3), np.uint8) * 255
    # c = 30
    # for contour in contours:
    #     cv2.fillConvexPoly(result_image, contour, (c, c, c))
    #     c += 2
    # cv2.imshow('img-windows', result_image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()
