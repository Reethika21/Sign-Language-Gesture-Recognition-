from keras.models import model_from_json
import cv2
import numpy as np

# Load the JSON file containing the model architecture
json_file_path = r"C:\Users\HP\Desktop\sign language\signlanguagedetectionmodel48x48.json"
with open(json_file_path, "r") as json_file:
    model_json = json_file.read()

# Reconstruct the Keras model from the JSON string
model = model_from_json(model_json)

# Load the model weights
model.load_weights(r"C:\Users\HP\Desktop\sign language\signlanguagedetectionmodel48x48.h5")

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape to match model input
    return feature / 255.0  # Normalize the pixel values

# Initialize webcam capture
cap = cv2.VideoCapture(0)
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Blank']

# Start video capture loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break  # Exit if frame not grabbed

    # Draw a rectangle on the frame
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    cropframe = frame[40:300, 0:300]  # Crop the region of interest
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    cropframe = cv2.resize(cropframe, (48, 48))  # Resize to the input size
    cropframe = extract_features(cropframe)  # Preprocess the image

    # Make prediction
    pred = model.predict(cropframe)
    prediction_label = label[pred.argmax()]

    # Draw prediction label on the frame
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'Blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)  # Get prediction accuracy
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("output", frame)  # Display the frame
    if cv2.waitKey(27) & 0xFF == ord('q'):  # Exit loop on 'q' key press
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
