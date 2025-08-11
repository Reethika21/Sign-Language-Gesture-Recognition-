import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image

# Load the JSON file containing the model architecture
json_file_path = r"C:\Users\HP\Desktop\signlanguage\signlanguagedetectionmodel48x48.json"
with open(json_file_path, "r") as json_file:
    model_json = json_file.read()

# Reconstruct the Keras model from the JSON string
model = model_from_json(model_json)

# Load the model weights
model.load_weights(r"C:\Users\HP\Desktop\signlanguage\signlanguagedetectionmodel48x48.h5")

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape to match model input
    return feature / 255.0  # Normalize the pixel values

# List of labels (sign language letters A-Z and Blank)
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Blank']

# Streamlit interface
st.title("Real-time Sign Language Recognition")

# OpenCV for capturing video from webcam
cap = cv2.VideoCapture(0)

# Streamlit elements for displaying information
frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# Display the webcam feed and predictions in real-time
while True:
    ret, frame = cap.read()
    
    if not ret:
        st.write("Error: Failed to capture frame.")
        break
    
    # Draw a rectangle on the frame for the region of interest
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    cropframe = frame[40:300, 0:300]  # Crop the region of interest
    
    # Convert to grayscale and resize the cropped frame
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    cropframe = cv2.resize(cropframe, (48, 48))  # Resize to the input size
    
    # Only process if the crop contains significant content (not all black/blank)
    if np.sum(cropframe) > 1000:  # Adjust this threshold for sensitivity
        # Preprocess the image
        cropframe = extract_features(cropframe)

        # Make prediction
        pred = model.predict(cropframe)
        prediction_label = label[pred.argmax()]

        # Prediction accuracy
        accuracy = "{:.2f}".format(np.max(pred) * 100)
        
        # Display prediction and accuracy
        prediction_placeholder.text(f"Prediction: {prediction_label} ({accuracy}%)")
        
        # Draw prediction label on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
        if prediction_label == 'Blank':
            cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'{prediction_label} {accuracy}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        # If nothing in the frame, display a default message
        prediction_placeholder.text("No hand detected. Place your hand in front of the camera.")
        
    # Convert frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame using Streamlit
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    # Exit the loop if 'q' is pressed (pressing 'q' closes the video stream)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
