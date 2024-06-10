import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time

# Load the pre-trained model
model = load_model(r"\Users\prash\OneDrive\Desktop\jay_projecf\jay_projecf\jayanthModel.h5")
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Video capture
cap = cv2.VideoCapture(0)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:  # Process all hands detected
            x_min, y_min = np.Inf, np.Inf
            x_max, y_max = -np.Inf, -np.Inf
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Check if the region is valid
            if x_min < x_max and y_min < y_max:
                hand_region = frame[y_min:y_max, x_min:x_max]
                if hand_region.size != 0:  # Ensure the region is not empty
                    # Resize hand region to the model's input size (assuming 28x28)
                    hand_resized = cv2.resize(hand_region, (28, 28))
                    hand_resized = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2GRAY)
                    hand_resized = hand_resized / 255.0
                    hand_resized = np.expand_dims(hand_resized, axis=0)
                    hand_resized = np.expand_dims(hand_resized, axis=-1)
                    
                    # Predict character using the model
                    prediction = model.predict(hand_resized)
                    predicted_class = np.argmax(prediction)
                    predicted_letter = letterpred[predicted_class]
                    
                    # Display the predicted letter on the frame
                    cv2.putText(frame, predicted_letter, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
