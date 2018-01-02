# Skin detection

Using the [Skin Segmentation Dataset](https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation), a decision tree is
trained to recognize skin in pictures. SVM is also included, but the model takes too much to train (and decision tree's
performance is very good already).

## Usage
```
python detection_by_mlalgorithm.py <target image> <algorithm> <filename to save model or load a previously trained
model> <use_hsv>
```

Where:
- target image: The image we want to detect skin on.
- algorithm: Either "tree" or "SVM". If a model was not found on the specified path, a new model is trained using the
selected algorithm.
- filename: Set the path of a previous trained model. If not found, a new model will be trained and saved on this path.
- use_hsv: "1" if you want to use the HSV color space. "0" if you want to use RGB.

Results vary depending on the color space, but RGB tends to give better detections.

## Flow

The program first uses a [face_recognition model](https://github.com/ageitgey/face_recognition), then extracts the face
and uses it as input to the skin detection part. Once the skin has been detected, the program creates an image with
similar sizes as the original and sets all pixels to black except where skin was detected on the extracted faces.

## Requirements

- Install [dlib](https://github.com/davisking/dlib)
- Download the face recognition models from  git+https://github.com/ageitgey/face_recognition_models


## TODO

The main idea is to detect persons' complete bodies and use the skin detection algorithm to reduce false positives.
The current project just needs skin recognition on faces, but detection on the whole body remains as a task to
be done.