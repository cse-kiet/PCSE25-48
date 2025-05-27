# Library imports


import streamlit as st

# ✅ FIRST command must be set_page_config
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# ✅ Add background color and style
st.markdown(
    """
    <style>
    body {
        background-color: #90EE90; /* Light green */
    }
    .centered-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #004d00; /* Dark green text */
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Centered title using HTML
st.markdown('<div class="centered-title">🌿 WELCOME: Plant Disease Detection App</div>', unsafe_allow_html=True)

# Continue with rest of your app...




import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
model = load_model('plant_disease_model.h5')
                    
# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type = "jpg")
submit = st.button('predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert image to 4 Dimension
        opencv_image.shape = (1, 256, 256, 3)
        
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ " leaf with " +  result.split('-')[1]))
