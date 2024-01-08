import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from tempfile import NamedTemporaryFile

# Custom objects for loading the model
custom_objects = {'optimizer_experimental.Optimizer': tf.optimizers.Adam}

# Set Streamlit page configuration
st.set_page_config(page_icon="fire_favicon.ico")

# Sidebar section with "About Me" information
st.sidebar.title("About Me")

# Sidebar customization
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

# Profile picture customization
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

# Sidebar content
st.sidebar.image("your_profile_picture.jpg", use_column_width=True, output_format='PNG')
st.sidebar.write("Hi! I'm Sarang, the creator of this forest fire detection system. "
                 "I'm glad that you're finding this application useful. "
                 "If you have any questions or would like to connect, feel free to reach out to me.")
st.sidebar.write("Email: bathalapalli9920@gmail.com")
st.sidebar.write("Social Media: [LinkedIn](https://www.linkedin.com/in/b-sarang-8b5b20217/), "
                 "[Instagram](https://www.instagram.com/sarrang9/)")

# Load the model
model_filename = "mobilenetV2_P150623.h5"
model_path = os.path.join(os.getcwd(), model_filename)
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Preprocess input image
def preprocess_image(img_path):
    """
    Preprocess the input image.

    Parameters:
    - img_path (str): Path to the image file.

    Returns:
    - preprocessed_img (numpy.ndarray): Preprocessed image array.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)
    return preprocessed_img

# Make individual predictions
def make_prediction(img_path):
    """
    Make predictions for the input image.

    Parameters:
    - img_path (str): Path to the image file.

    Returns:
    - predicted_class (str): Predicted class ('no fire' or 'fire').
    - confidence (float): Confidence level of the prediction.
    """
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    class_names = ['no fire', 'fire']
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

def main():
    """
    Streamlit application main function.
    """
    st.title('A Deep Learning based approach to detecting forest fires')

    # Custom CSS styles for background and text color
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
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(uploaded_file.getvalue())

        image_path = temp_file.name
        predicted_class, confidence = make_prediction(image_path)

        if predicted_class == 'fire':
            st.markdown(
                f'<p class="predicted-class predicted-class-fire" style="font-size: 60px;">🔥 {predicted_class}</p>',
                unsafe_allow_html=True)
            st.warning("Fire Detected!")
        else:
            st.markdown(
                f'<p class="predicted-class predicted-class-no-fire" style="font-size: 60px;">🍃 {predicted_class}</p>',
                unsafe_allow_html=True)

        st.subheader('Result')
        st.image(image_path, use_column_width=True, width=300, caption='Uploaded Image')

if __name__ == '__main__':
    main()
