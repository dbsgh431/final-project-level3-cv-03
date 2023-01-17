import io
import requests
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")

def main():
    st.title("Pothole Classification Model")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")

        files = [
            ('files', (uploaded_file.name, image_bytes,
                       uploaded_file.type))
        ]
        response = requests.post("http://localhost:8001/predict", files=files)
        label = response.json()["result"]
        st.write(f'label is {label}')

main()