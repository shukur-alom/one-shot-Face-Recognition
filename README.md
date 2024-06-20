# One-Shot Learning for Real-Time Face Recognition

This project implements a Siamese Network for face recognition using live video feed. The Siamese Network is trained to distinguish between different faces by learning to differentiate between pairs of images.

[![Watch the video](https://i.sstatic.net/Vp2cE.png)](https://github.com/shukur-alom/one-shot-Face-Recognition/blob/main/Media/Demo.mp4)


## Table of Contents
- [Introduction](#introduction)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Real-Time Application](#real-time-application)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#Contact)



## Introduction

A Siamese Network consists of two identical neural networks with shared weights, which are used to find the similarity between two inputs. This project applies this concept to face recognition, allowing for real-time identification of individuals using a live video feed.

![Model](https://github.com/shukur-alom/one-shot-Face-Recognition/blob/main/Media/0_lgjFPlTjPjiW4ziu-transformed.png)

## Dataset Preparation

1. **Extract Faces**: Use face detection techniques (YOLOV8) to extract faces from a collection of images or video frames.
2. **Create a New Dataset**: Organize the extracted face images into a structured dataset. The directory structure should be as follows:
    ```
    dataset/
    ├── person1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── person2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
    ```

## Model Training

1. **Preprocess Images**: Convert images to BGR, resize them to a fixed size, and normalize pixel values.
2. **Create Pairs**: Generate positive and negative pairs of images. Positive pairs are two images of the same person, and negative pairs are two images of different people.
3. **Data Augmentation**: Apply data augmentation techniques such as random rotations, shifts, flips, and zooms to increase the diversity of the training dataset and improve model robustness.
4. **Train Siamese Network**: Train the Siamese Network using the generated pairs. The network learns to output a similarity score indicating whether two images belong to the same person.

## Real-Time Application

1. **Capture Video**: Use a webcam or any video capturing device to get a live feed.
2. **Preprocess Frame**: For each frame, detect faces and preprocess the face images.
3. **Compare Faces**: Compare the detected faces with known faces using the trained Siamese Network to determine if they match any known individual.
4. **Display Results**: Display the video feed with recognition results overlayed on the detected faces.

## Dependencies

- Python 3.9+
- TensorFlow
- ultralytics
- OpenCV
- NumPy

You can install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare Dataset**: Ensure your dataset is organized as described in the Dataset Preparation section.
2. **Train Model**: Run the training [NoteBook](https://github.com/shukur-alom/one-shot-Face-Recognition/blob/main/face-similarity-siamese-model.ipynb) to train the Siamese Network.
3. **Run Real-Time Recognition**:Use the following command to start the real-time face recognition.
Before run change image path.
```bash
python main.py
```

## Contributing

**Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.**

## Contact

For any questions or inquiries, you can reach me on LinkedIn:
[Shukur Alam](https://github.com/shukur-alom/one-shot-Face-Recognition/blob/main/face-similarity-siamese-model.ipynb)