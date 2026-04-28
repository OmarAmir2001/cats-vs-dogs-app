# ui/app.py
import streamlit as st
from PIL import Image
import requests

# Where our API lives — change this if you deploy elsewhere
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Cats vs Dogs", page_icon="🐾", layout="centered")

st.title("🐱🐶 Cats vs Dogs Classifier")
st.write("Upload a photo of a cat or a dog and we'll tell you which it is.")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your uploaded image", width=400)

    # Button to trigger the API call
    if st.button("Classify"):
        with st.spinner("Asking the model..."):
            # Reset the file pointer so we can re-read the bytes
            uploaded_file.seek(0)

            # POST the file to the API
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            confidence = result['confidence']

            # Show the result in a big, friendly way
            emoji = "🐱" if prediction == "cat" else "🐶"
            st.success(f"{emoji} It's a **{prediction}**! ({confidence*100:.1f}% confident)")

            # Show the probability bars for both classes
            st.write("### Probabilities")
            for class_name, prob in result['probabilities'].items():
                st.write(f"**{class_name}**")
                st.progress(prob)
        else:
            st.error(f"Something went wrong: {response.status_code}")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a ResNet18 model fine-tuned on cat and dog photos from Kaggle. "
        "The classifier reaches ~99% accuracy on a held-out test set."
    )
    st.write("Built with PyTorch, FastAPI, and Streamlit.")