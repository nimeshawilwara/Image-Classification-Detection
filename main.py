import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

model = InceptionV3(weights='imagenet')

def classify_image(image):
    image = image.convert("RGB")
    target_size = (299, 299)
    image = image.resize(target_size)

    img_array = kimage.img_to_array(image)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
  
    decoded_predictions = decode_predictions(predictions)[0]
    return decoded_predictions
  
def main():
    st.sidebar.title("Hi There ... !")
    st.sidebar.title("Image Classification App")
    st.sidebar.write("This is Image Classification Web App. Please choose your Image and after the few minites you can show the Result.")
    
    st.title("Image Classification Application")

    uploaded_file = st.file_uploader("Select an Your Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="You Uploaded Image Here...", use_column_width=True)

        predictions = classify_image(image)

        st.subheader("Classification OutPuts/Results:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"{i + 1}: {label} ({score:.2f})")

if __name__ == "__main__":
    main()
