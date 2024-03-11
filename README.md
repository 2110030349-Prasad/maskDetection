# Face Mask Detection System

This Python script demonstrates a face mask detection system using machine learning, particularly the Support Vector Machine (SVM) algorithm.

## Description

The script performs the following tasks:

1. **Import necessary libraries**: 
    - `numpy`: For numerical computations.
    - `cv2`: OpenCV library for image processing.

2. **Data Preprocessing**:
    - Load the preprocessed face image datasets with and without masks.
    - Reshape the datasets to have 200 samples of 50x50x3 pixels for each class (with and without masks).
    - Concatenate the datasets into a single array `X`.
    - Create labels for the datasets: 0 for images with masks and 1 for images without masks.
    - Define a dictionary `name` to map labels to their corresponding classes.

3. **Model Training**:
    - Import SVM classifier and accuracy metric from scikit-learn.
    - Split the data into training and testing sets.
    - Perform Principle Component Analysis (PCA) to reduce the dimensionality of the feature space.
    - Train the SVM classifier on the training data.
    - Predict the labels of the test data.
    - Calculate and print the accuracy of the model.

4. **Real-time Detection**:
    - Set up the webcam capture using OpenCV.
    - Define a loop for real-time face detection and mask prediction:
        - Detect faces using the Haar cascade classifier.
        - Draw rectangles around detected faces.
        - Resize and preprocess each face image.
        - Predict whether the face is with or without a mask using the trained SVM model.
        - Display the prediction label on the detected face.
        - Break the loop if the 'Esc' key is pressed.
    - Release the webcam and close all OpenCV windows.


## Dependencies

- Python 3.x
- NumPy
- OpenCV (cv2)
- scikit-learn
