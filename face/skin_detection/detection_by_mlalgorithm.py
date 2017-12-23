# Trains a classifier based on the Skin Segmentation Dataset (https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
# The decision tree algorithm trains very fast. SVM takes a long time (22 minutes on my machine).
# ==> RGB gives better results so far.

# Added face recognition temporarily.
# Source: https://github.com/ageitgey/face_recognition
# In order to make this work, you'll need to follow the installation instructions there.
# In short:
# - Install dlib
# - Download the face recognition models from git+https://github.com/ageitgey/face_recognition_models

# The main idea is to detect persons' complete bodies and then use the skin detection algorithm
# to reduce false positives. Face recognition is just a sample of that idea. As soon as I get the
# proper datasets I'm planning to use them for detection.

import numpy as np
import cv2
import time
import face_recognition
import os

from sklearn import tree
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sys import argv

IMAGE_EXTENSIONS = ['.jpg', '.png' '.bmp']


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


def get_model(filename):
    if os.path.isfile(filename):
        return load_model(filename)
    else:
        return None


def save_model(model, filename):
    joblib.dump(model, filename)
    print("==> Model saved in {:s}".format(filename))


def load_model(filename):
    print("==> Loading model in file {:s}".format(filename))
    return joblib.load(filename)


def maybe_is_image(filename):
    return True if filename.lower()[-4:] in IMAGE_EXTENSIONS else False


def apply_to_image(path, classifier, use_hsv=True):

    img = cv2.imread(path)

    # Detect face
    # print(img)
    # img = np.roll(img, 1, axis=-1)
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")

    for face_location in face_locations:
        top, right, bottom, left = face_location

        face_img = img[top:bottom, left:right]
        data = np.reshape(face_img, (face_img.shape[0] * face_img.shape[1], 3))

        if use_hsv:
            data = bgr_to_hsv(data)

        predicted_labels = classifier.predict(data)

        img_labels = np.reshape(predicted_labels, (face_img.shape[0], face_img.shape[1], 1))

        new_image = np.zeros_like(img)

        extension = ''
        if use_hsv:
            new_image[top:bottom, left:right] = (-(img_labels-1)+1)*255
            extension = '_HSV.'
        else:
            new_image[top:bottom, left:right] = (-(img_labels - 1) + 1) * 255
            extension = '_RGB.'

        cv2.imwrite(path[:-3] + extension + path[-3:],
                    new_image)  # from [1 2] to [0 255]


def main(argv):
    image = argv[1]
    algorithm = argv[2]
    classifier_filename = argv[3]
    use_hsv = True if len(argv) == 5 and argv[4] == '1' else False

    classifier = get_model(classifier_filename)

    if classifier is None:
        data, labels = read_data()
        classifier = train(data, labels, algorithm=algorithm, use_hsv=use_hsv)
        save_model(classifier, classifier_filename)

    apply_to_image(image, classifier, use_hsv=use_hsv)


if __name__ == '__main__':
    main(argv)