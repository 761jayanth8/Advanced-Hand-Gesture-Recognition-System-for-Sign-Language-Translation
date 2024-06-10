# Hand Gesture Recognition System

This project implements a real-time hand gesture recognition system using TensorFlow, OpenCV, and MediaPipe. The system captures video frames from the webcam, processes them to detect hands, and uses a pre-trained model to predict hand gestures. The predicted gestures are then displayed on the video feed.

## Table of Contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Running the Application](#running-the-application)
4. [Model](#model)
5. [How It Works](#how-it-works)
6. [Troubleshooting](#troubleshooting)
7. [Credits](#credits)
8. [Model Requests](#model-requests)

## Requirements

Before running the code, ensure you have the following packages installed:

- TensorFlow
- OpenCV
- MediaPipe
- Keras
- NumPy

You can install these packages using pip:

```bash
pip install tensorflow opencv-python mediapipe keras numpy
```

## Setup

1. **Clone the Repository**:
   
   Clone the repository to your local machine.

   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. **Model File**:

   Ensure you have the pre-trained model file `jayanthModel.h5` in the appropriate directory. Update the path to the model file in the code if necessary.

   ```python
   model = load_model(r"\Users\prash\OneDrive\Desktop\jay_projecf\jay_projecf\jayanthModel.h5")
   ```

3. **Run the Application**:

   Execute the Python script to start the hand gesture recognition system.

   ```bash
   python hand_gesture_recognition.py
   ```

## Running the Application

1. **Start Video Capture**:
   
   The application captures video from the default webcam. Ensure your webcam is connected and working properly.

2. **Hand Gesture Prediction**:
   
   The application processes each frame to detect hands and predicts the corresponding gesture using the pre-trained model.

3. **Display Predictions**:
   
   The predicted gesture is displayed on the video feed in real-time.

4. **Exit**:
   
   Press the `q` key to exit the application.

## Model

The model used in this project is a pre-trained Convolutional Neural Network (CNN) saved as `jayanthModel.h5`. The model is trained to recognize various hand gestures corresponding to letters in the alphabet.

The labels for the predictions are:

```python
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
```

## How It Works

1. **Video Frame Capture**:
   
   Captures frames from the webcam using OpenCV.

2. **Hand Detection**:
   
   Uses MediaPipe's Hands module to detect hand landmarks in the frame.

3. **Region of Interest Extraction**:
   
   Extracts the bounding box around the detected hand and preprocesses the region for prediction.

4. **Gesture Prediction**:
   
   Resizes the hand region to 28x28 pixels, converts it to grayscale, normalizes pixel values, and feeds it into the CNN model to get the predicted gesture.

5. **Display Prediction**:
   
   Displays the predicted gesture on the video frame using OpenCV.

## Troubleshooting

- **No Frame Capture**:
  Ensure your webcam is connected and selected correctly. Check the `cap = cv2.VideoCapture(0)` line to ensure the correct device index is used.

- **Invalid Model Path**:
  Ensure the path to the `jayanthModel.h5` file is correct.

- **Dependencies**:
  Ensure all required packages are installed. Use `pip install` to install any missing packages.

## Credits

This project is developed by Jayanth Rahul as part of a guitar learning system. The code leverages TensorFlow, OpenCV, and MediaPipe to achieve real-time hand gesture recognition.

## Model Requests

If you need the `jayanthModel.h5` file for this project, please reach out, and I will provide it to you. Feel free to contact me for any questions or issues regarding this project!
