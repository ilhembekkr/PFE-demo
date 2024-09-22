# Importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
from PIL import Image
from core.models.vgg import VGGClassificationModel


# Define a function to load the model into session state
def load_model(selected_model):
    with st.spinner(f'Loading {selected_model} model...'):
        # Simulate loading different models based on the selectbox selection
        if selected_model == "VGG-16":
            st.session_state['model'] = VGGClassificationModel()
        else:
            # Implement loading logic for other models
            st.session_state['model'] = VGGClassificationModel()  # Placeholder
        st.session_state['model_name'] = selected_model
        st.success(f"Model {selected_model} loaded successfully!", icon='🚀')

st.write("# Differentiable Joint DA-NAS for Image Classification")

# Model options
model_options = ["da-nas-10", "da-nas-cifar100"]  
selected_model = st.selectbox("Choose a model:", model_options)

# Load a new model whenever the selected model changes
if 'model_name' not in st.session_state or st.session_state['model_name'] != selected_model:
    load_model(selected_model)

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Classifying with {selected_model} ...")
    
    # Use the pre-loaded model to predict
    predicted_class = st.session_state['model'].predict(image)
    st.write(f"Prediction: {predicted_class}")
