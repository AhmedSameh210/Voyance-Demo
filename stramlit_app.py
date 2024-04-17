import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

logo_path = 'voyance logo.png'
model_path = 'model.h5'

# Function to load and preprocess image
def preprocess_image(image):
    resized_image = cv2.resize(image, (48, 48))
    processed_image = resized_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

# Main function for Streamlit app
def main():
    # Set page title and configure page layout
    st.set_page_config(page_title='Voyance Health', layout='wide', initial_sidebar_state='collapsed')

    # Inject custom CSS for black background
    st.markdown(
        """
        <style>
        body {
            background-color: black;
            color: white; /* Set text color to white for contrast */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set app title and logo
    st.title('Voyance Health')

    # Display logo with left alignment
    st.sidebar.image(logo_path, use_column_width=True)

    # Display file uploader widget
    uploaded_file = st.file_uploader("Choose an image to make a Medical Decision", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image from file uploader
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Load pre-trained model
        model = load_model()
        
        # Make predictions
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        if predicted_class == 0 :
            out="Healthy Person"
        elif predicted_class ==1:
            out="Type 1 Disease Person"
        elif predicted_class == 2 :
            out="Type 2 Disease Person"
        else :
             out="error in prediction"
        
        # Display prediction
        st.write(f"Predicted Medical Decision:  {out}")

# Run the main function
if __name__ == '__main__':
    main()
