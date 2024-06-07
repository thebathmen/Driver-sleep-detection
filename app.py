import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("drowiness_new6.h5")

# Labels for the output classes
labels_new = ["yawn", "no_yawn", "Closed", "Open"]

# Function to prepare the image for prediction
def prepare(face_image):
    IMG_SIZE = 145
    img_array = face_image / 255.0
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Function to handle image selection and prediction
def predict_faces_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    sleeping_count = 0
    prediction_text = ""

    # For each detected face
    for i, (x, y, w, h) in enumerate(faces):
        face_image = image[y:y+h, x:x+w]
        
        # Prepare the face image for prediction
        prepared_image = prepare(face_image)
        
        # Make a prediction
        predictions = model.predict(prepared_image)
        
        # Extract the predicted class for classification
        classification_prediction = predictions[0]
        predicted_class_index = np.argmax(classification_prediction)
        predicted_label = labels_new[predicted_class_index]
        
        # Extract the predicted age (assuming second part of prediction is age)
        predicted_age = predictions[1][0][0]
        
        # Check if the person is sleeping
        if predicted_label in ["Closed", "yawn"]:
            sleep_status = "Sleeping"
            sleeping_count += 1
        else:
            sleep_status = "Awake"
        
        prediction_text += f"Person {i+1}: {sleep_status}, Age: {predicted_age:.2f}\n"
    
    # Show the results in a message box
    prediction_text += f"\nTotal sleeping persons: {sleeping_count}"
    return prediction_text, sleeping_count

# Function to process video frames and make predictions
def process_video(video_file):
    vid = cv2.VideoCapture(video_file)
    frame_count = 0
    total_sleeping = 0
    
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:  # Process one frame per second
            prediction_text, sleeping_count = predict_faces_in_image(frame)
            total_sleeping += sleeping_count
    
    vid.release()
    return total_sleeping

# Streamlit interface
st.title("Driver Drowsiness Detection")

# Image prediction
st.header("Image Prediction")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        # Convert the file to an OpenCV image
        image = Image.open(uploaded_image)
        image = np.array(image)
        
        # Check if the image was converted correctly
        st.image(image, caption='Uploaded Image', use_column_width=True, channels="RGB")
        
        # Predict and display results
        prediction_text, _ = predict_faces_in_image(image)
        st.text(prediction_text)
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Video prediction
st.header("Video Prediction")
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        # Process the video and display results
        total_sleeping = process_video("temp_video.mp4")
        st.text(f"Total sleeping persons detected in video: {total_sleeping}")
    except Exception as e:
        st.error(f"Error processing video: {e}")
