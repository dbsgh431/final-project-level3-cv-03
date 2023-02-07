import os
import io
from PIL import Image
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events

from google.cloud import storage

# GCP 사용을 위한 서비스 계정 키 _ 환경변수에 키 경로를 설정
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/opt/ml/essential_config/aitech4-finalproject-0000-b7e0b3c932a1.json'

# GCP 및 버킷 설정
storage_client = storage.Client()
before_bucket = storage_client.bucket('detected_potholes')
after_bucket = storage_client.bucket('repaired_potholes')

# 촬영된 이미지가 저장되는 버킷
# 시각화를 위한 구조화
need_to_repair_potholes = storage_client.list_blobs('detected_potholes')
metadata_dict = {}
for data in need_to_repair_potholes:
    metadata_dict[data.name] = data.metadata
df = pd.DataFrame(metadata_dict).T

# 데이터의 위도/경도
latitude, longitude = [], []
if len(df) != 0:
    latitude = df['lat'].apply(lambda x : float(x)).values
    longitude = df['lon'].apply(lambda x : float(x)).values

# 보수완료 데이터를 옮기는 함수
def tranf_blob(obj, bk1, bk2):
    bk2.copy_blob(obj, bk2, obj.name)
    bk1.delete_blob(obj.name)


# Streamlit Page
st.set_page_config(layout="wide")
st.title('Pothole Map')
map_layout, img_layout, empty2 = st.columns([1.0,0.4,0.1])

with map_layout:
    fig = px.scatter_mapbox(lat=latitude, lon=longitude, mapbox_style="carto-positron",
                            center={'lat':37.5348623, 'lon':127.0356773}, zoom=11, width=1200, height=1000)
    selected_point = plotly_events(fig, click_event=True, hover_event=False)
    if st.button('Refresh'):
        selected_point = []
    
with img_layout:
    if len(selected_point) != 0:
        st.header('Image Preview')
        selected_blob = before_bucket.get_blob(df.index[selected_point[0]['pointIndex']])
        selected_img = Image.open(io.BytesIO(selected_blob.download_as_bytes()))
        st.image(selected_img)
        st.write(selected_blob.name)
        
        # 보수완료 시 데이터 이동
        if st.button('보수완료'):
            tranf_blob(selected_blob, before_bucket, after_bucket)
            selected_point = []