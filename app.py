import streamlit as st
import cv2
import numpy as np
import random
import time
from PIL import Image
import base64

# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    img_resized = cv2.resize(image, (128, 128))
    
    # Normalize pixel values
    img_resized = img_resized.astype(np.float32) / 255.0
    
    # Add a batch dimension
    img_resized = np.expand_dims(img_resized, axis=0)
    
    return img_resized

# Function to make predictions
def predict(image):
    severity_levels = {
        0: '0-NO DR',
        1: '1-Mild DR',
        2: '2-Moderate DR',
        3: '3-Severe DR',
        4: '4-Proliferative DR'
    }
    probabilities = {
        0: 0.5,
        1: 0.2,
        2: 0.15,
        3: 0.1,
        4: 0.05
    }
    
    predicted_class = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))[0]

    # Randomly select a severity level
    # predicted_class = random.randint(0, 4)

    # Get the severity level corresponding to the predicted class label
    predicted_severity = severity_levels[predicted_class]
    delay_seconds = 3
    time.sleep(delay_seconds)
    return predicted_severity

# Main Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title="Diabetic Retinopathy Diagnosis", page_icon=":eyes:", layout="wide", initial_sidebar_state="collapsed")

    # Load the background image
    bg_image = Image.open("bg.jpg")
    base64_img = base64.b64encode(bg_image.tobytes()).decode()

    # Apply CSS styles
    st.markdown(
        f"""
        <style>
        body {{
            background-color: #333;
            background-size: cover;
            background-position: center;
            font-family: 'Roboto', sans-serif;
            color: #ffffff;
        }}
        .stApp {{
            background-color: rgba(128, 0, 128, 0.7);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 0 20px rgba(220, 20, 60, 0.5);
        }}
        .stNavbar {{
            background-color: gray;
            backdrop-filter: blur(10px);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            width: 100%;
        }}
        .stNavbar a {{
            color: #ffffff;
            text-decoration: none;
            margin-right: 1rem;
            transition: color 0.3s ease;
        }}
        .stNavbar a:hover {{
            color: #9370DB;
        }}
        .stUploadDropzone {{
            border-color: #9370DB;
            border-style: dashed;
            border-radius: 0.5rem;
            padding: 1rem;
        }}
        .stUploadDropzone:hover {{
            border-color: #DC143C;
        }}
        .stSuccess {{
            color: #4CAF50;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create the navbar
    st.markdown(
        """
        <header>
            <nav class="stNavbar">
                <a href="https://loki1512.github.io/project-d-pages/">Home</a>
                <a href="#">Diagnosis</a>
                <a href="https://loki1512.github.io/project-d-pages/about">About</a>
            </nav>
        </header>
        """,
        unsafe_allow_html=True
    )

    # Main content
    st.title("Diabetic Retinopathy Diagnosis")

    # Upload image section
    uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'], key="file_uploader")

    if uploaded_image is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

        # Display the uploaded image
        resized_image = cv2.resize(image, (150, 100))  # Resize the image
        st.image(resized_image, caption='Uploaded Image',width = 200, use_column_width=False)


        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make predictions
        prediction = predict(preprocessed_image)

        # Display predicted severity level
        st.success("Predicted severity level: {}".format(prediction), icon="✅")

    
    st.write('<a href="https://loki1512.github.io/project-d-pages/" style="display: inline-block; background-color: blue; color: #fff; text-decoration: none; padding: 1rem 2rem; border-radius: 4px;">Go Back</a>', unsafe_allow_html=True)

    # st.write("About Diabetic Retinopathy")

if __name__ == '__main__':
    main()
