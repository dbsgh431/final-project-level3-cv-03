import io
import requests
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")

def main():
    st.title("Test Page _ Pothole Detection")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        coord = {'lat' : 38.0, 'lon' : 128.0}
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        upfile = {'files' : image_bytes}
        
        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")

        response = requests.post(f"http://localhost:30011/OD", files=upfile, data=coord)
        label = response.status_code
        st.write(label)
        st.write('Work Completed')
main()