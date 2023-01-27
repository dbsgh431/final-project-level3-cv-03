import io
import requests
from PIL import Image
import streamlit as st

work_corres = {'Image Classification' : 'IC', 'Object Detection' : 'OD'}
st.set_page_config(layout="wide")

def main():
    st.title("Pothole Classification Model")
    work = st.sidebar.selectbox("작업 선택", ("Image Classification", "Object Detection"))
    st.header(work)
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        upfile = {'files' : image_bytes}
        
        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")

        response = requests.post(f"http://localhost:30011/{work_corres[work]}", files=upfile)
        if work == 'Image Classification':
            label = response.json()['result']
            st.write(f'label is {label}')
        elif work == 'Object Detection':
            label = response.content
            result_img = Image.open(io.BytesIO(label))
            st.image(result_img, caption='Uploaded Image')
        
        st.write('Work Completed')
main()