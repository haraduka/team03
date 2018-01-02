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

## Examples

Original | Result
--- | --- |
![Example 1](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/Gisele_Bundchen2.jpg "Example 1")  | ![Example 1 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/Gisele_Bundchen2._RGB.jpg "Example 1 result")
![Example 2](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/barackobama.jpg "Example 2")  | ![Example 2 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/barackobama._RGB.jpg "Example 2 result")
![Example 3](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/biden.jpg "Example 3")  | ![Example 3 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/biden._RGB.jpg "Example 3 result")
![Example 4](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/candice_swanepoel.jpg "Example 4")  | ![Example 4 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/candice_swanepoel._RGB.jpg "Example 4 result")
![Example 5](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/hashimoto-kanna.jpg "Example 5")  | ![Example 5 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/hashimoto-kanna._RGB.jpg "Example 5 result")
![Example 6](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/hazelkreech.jpg "Example 6")  | ![Example 6 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/hazelkreech._RGB.jpg "Example 6 result")
![Example 7](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/kiritanimirei.jpg "Example 7")  | ![Example 7 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/kiritanimirei._RGB.jpg "Example 7 result")
![Example 8](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/mirandakerr.jpg "Example 8")  | ![Example 8 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/mirandakerr._RGB.jpg "Example 8 result")
![Example 9](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/obama1.jpg "Example 9")  | ![Example 9 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/obama1._RGB.jpg "Example 9 result")
![Example 10](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/obama2.jpg "Example 10")  | ![Example 10 result](https://github.com/OtsuboAtsushi/team03/blob/skin_detection/face/skin_detection/examples/obama2._RGB.jpg "Example 10 result")

## TODO

The main idea is to detect persons' complete bodies and use the skin detection algorithm to reduce false positives.
The current project just needs skin recognition on faces, but detection on the whole body remains as a task to
be done.