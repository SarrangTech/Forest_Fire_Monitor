import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pydub import AudioSegment
from pydub.playback import play
import pygame
import os
import time
from tempfile import NamedTemporaryFile

# from aboutme import show_about_me
custom_objects = {'optimizer_experimental.Optimizer': tf.optimizers.Adam}
st.set_page_config(page_icon="fire_favicon.ico")
# Add "About Me" sidebar section
    # Add "About Me" sidebar section
st.sidebar.title("About Me")

    # Adjust sidebar width
st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            width: 300px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Customize profile picture size
st.markdown(
        """
        <style>
        .profile-pic {
            display: block;
            margin: 0 auto;
            width: 150px;
            border-radius: 50%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.sidebar.image("your_profile_picture.jpg", use_column_width=True,
                     output_format='PNG')

st.sidebar.write("Hi! im Sarang, the creator of this forest fire detection system. Im glad that your finding this application useful, If you have any questions or would like to connect, feel free to reach out to me.")
st.sidebar.write("Email: bathalapalli9920@gmail.com")
st.sidebar.write("Social Media: [LinkedIn](https://www.linkedin.com/in/b-sarang-8b5b20217/), [Instagram](https://www.instagram.com/sarrang9/)")

# Load the model
model_filename = "mobilenetV2_P150623.h5"
model_path = os.path.join(os.getcwd(), model_filename)
# st.write(f"Model Path: {model_path}")  # Add this line to print the model path
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
# model = tf.keras.models.load_model("Forest_Fire_Monitor\\mobilenetV2_P150623.h5", custom_objects=custom_objects)
# Preprocess input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img

# Make individual predictions
def make_prediction(img_path):
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    # Modify the code below based on your specific class labels
    class_names = ['no fire', 'fire']  # Replace with your own class names
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

import time

def play_alarm():
    try:
        # Check if a valid audio device is available
        if pygame.mixer.get_init() is None:
            st.warning("No audio device found. Unable to play alarm.")
            return

        # Initialize pygame.mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load("Forest_Fire_Monitor\\alarm.mp3")

        # Play the audio
        pygame.mixer.music.play()

        # Display the audio player in Streamlit
        audio_file = open("Forest_Fire_Monitor\\alarm.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

        # Allow time for the audio to play
        time.sleep(5)

        # Stop the audio
        pygame.mixer.music.stop()
    except pygame.error as e:
        st.warning(f"Error playing alarm: {e}")




# def play_alarm():
#     pygame.mixer.init()
#     pygame.mixer.music.load("Forest_Fire_Monitor\\alarm.mp3")
#     pygame.mixer.music.play()

def main():

    st.title('A Deep Learning based approach to detecting forest fires')

    # Add custom CSS styles for background and text color
    st.markdown(
        """
        <style>
        body {
            background-color: #006400; /* dark green */
            color: #8B4513; /* dark brown */
        }
        .predicted-class {
            font-size: 50px;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }
        .predicted-class-fire {
            color: red;
        }
        .predicted-class-no-fire {
            color: green;
        }
        .confidence {
            font-size: 25px;
            font-weight: bold;
            margin-top: 20px;
            text-align: left;
            color: white;
        }
        .personal-info {
            font-size: 18px;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }
        .contact-info {
            margin-top: 10px;
            text-align: center;
        }
        .social-media-link {
            margin-right: 10px;
        }
        .about-me {
            margin-top: 20px;
            text-align: center;
        }
        .profile-pic {
            display: block;
            margin: 0 auto;
            width: 200px;
            border-radius: 50%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Create a temporary file
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(uploaded_file.getvalue())

        # Get the temporary file path
        image_path = temp_file.name

        # Make prediction
        predicted_class, confidence = make_prediction(image_path)
    
    # ...
    if predicted_class == 'fire':
        st.markdown('<style>@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0; } 100% { opacity: 1; } }</style>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: red; font-size: 24px; animation: blink 1s linear infinite;">FIRE DETECTED!</p>', unsafe_allow_html=True)

        # Display the result
        st.subheader('Result')

        # Display the image
        st.image(image_path, use_column_width=True, width=300, caption='Uploaded Image')

        # Display predicted class and confidence
    if predicted_class == 'fire':
        st.markdown(f'<p class="predicted-class predicted-class-fire">{predicted_class}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="predicted-class predicted-class-no-fire">{predicted_class}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="confidence">Confidence: {confidence:.2f}%</p>', unsafe_allow_html=True)
    # if predicted_class == 'fire':
    #         # Play alarm sound
    #     play_alarm()
    if st.button('Disable Alarm'):
        pygame.mixer.music.stop()
        

if __name__ == '__main__':
    main()
