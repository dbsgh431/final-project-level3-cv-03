import cv2
import streamlit as st
import requests
import time,io
import threading
import requests, json
from streamlit.script_run_context import get_script_run_ctx, add_script_run_ctx

st.set_page_config(layout="wide")

st.title("Webcam Live Feed")

numbers = st.empty()

def current_location():
    here_req = requests.get("http://www.geoplugin.net/json.gp")

    if (here_req.status_code != 200):
        print("현재좌표를 불러올 수 없음")
    else:
        location = json.loads(here_req.text)
        crd = {"lat": str(location["geoplugin_latitude"]), "lng": str(location["geoplugin_longitude"])}
 
    return crd

    
def inference(byte_im):
    files = [('files', byte_im)]
    crd = current_location()
    # loc = {"lat" : crd['lat'], "lng":crd['lng']}
    #response = requests.post('http://118.67.129.236:30011/OD', data=loc, files=files)
    response = requests.post("http://49.50.175.25:30001/predict", files=files)
    print(crd)
    label = response.json()["result"]   
    with numbers.container():
        st.write(f'label is {label}')

run = st.checkbox('동의 및 실행')
FRAME_WINDOW = st.image([])
txt =  '''
수집한 자료는 ...
'''
x= st.info(txt)
delta = 0
previous = time.time()
while run:
    if x:
        x.empty()
    camera = cv2.VideoCapture('http://192.168.10.103:8080/video')
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    current = time.time()
    delta += current - previous
    previous = current

    if delta > 3.0:
        is_success, im_buf_arr = cv2.imencode(".jpg", frame)
        io_buf = io.BytesIO(im_buf_arr)
        byte_im = io_buf.getvalue()

        byte_im = [byte_im]
        
        thread = threading.Thread(target=inference, args=byte_im)
        add_script_run_ctx(thread)
        thread.start()
        #inference(files)
        delta = 0
else:
    st.write('Stopped')
