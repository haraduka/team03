# Trains a classifier based on the Skin Segmentation Dataset (https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
# The decision tree algorithm trains very fast. SVM takes a long time (22 minutes on my machine).

import numpy as np
import cv2
import time

from sklearn import tree
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sys import argv


def read_data():
    # Data in format [B G R Label]
    data = np.genfromtxt('skin_segmentation_dataset/Skin_NonSkin.txt', dtype=np.int32)

    labels = data[:, 3]
    data = data[:, 0:3]

    return data, labels


def bgr_to_hsv(bgr):
    bgr = np.reshape(bgr, (bgr.shape[0], 1, 3))
    hsv = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv = np.reshape(hsv, (hsv.shape[0], 3))

    return hsv


def train(data, labels, algorithm, use_hsv):
    if use_hsv:
        data = bgr_to_hsv(data)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20, random_state=8943)

    if algorithm == 'tree':
        classifier = tree.DecisionTreeClassifier(criterion='entropy')
    else:
        classifier = SVC()

    print("*** Training ***")
    start_time = time.time()
    classifier = classifier.fit(train_data, train_labels)
    end_time = time.time()
    print("+++ Complete! Training time: {:.5f} minutes".format((end_time - start_time) / 60))

    return classifier


def apply_to_image(path, result_image, algorithm='tree', use_hsv=True):
    data, labels = read_data()
    classifier = train(data, labels, algorithm, use_hsv)

    img = cv2.imread(path)
    data = np.reshape(img, (img.shape[0]*img.shape[1], 3))

    if use_hsv:
        data = bgr_to_hsv(data)

    predicted_labels = classifier.predict(data)

    img_labels = np.reshape(predicted_labels, (img.shape[0], img.shape[1], 1))

    if use_hsv:
        cv2.imwrite(result_image[:-3] + '_HSV.' + result_image[-3:],
                    ((-(img_labels-1)+1)*255))  # from [1 2] to [0 255]
    else:
        cv2.imwrite(result_image[:-3] + '_RGB.' + result_image[-3:],
                    ((-(img_labels - 1) + 1) * 255))  # from [1 2] to [0 255]


def main(argv):
    image = argv[1]
    result_image = argv[2]
    algorithm = argv[3]

    apply_to_image(image, result_image, algorithm, True)

if __name__ == '__main__':
    main(argv)