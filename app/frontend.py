import os
from io import BytesIO
from PIL import Image
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events

from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/opt/ml/essential_config/aitech4-finalproject-0000-b7e0b3c932a1.json'
storage_client = storage.Client()

before_bucket = storage_client.bucket('detected_potholes')
after_bucket = storage_client.bucket('repaired_potholes')

# 중복은 일단 무시, 주기적 업데이트 무시, 보수 완료 시 버킷 이동
need_to_repair_potholes = storage_client.list_blobs('detected_potholes')
metadata_dict = {}
for data in need_to_repair_potholes:
    metadata_dict[data.name] = data.metadata
df = pd.DataFrame(metadata_dict).T

latitude = df['lat'].apply(lambda x : float(x)).values
longitude = df['lon'].apply(lambda x : float(x)).values

# 보수완료 데이터를 옮기는 함수
def tranf_blob(obj, bk1, bk2):
    bk2.copy_blob(obj, bk2, obj.name)
    bk1.delete_blob(obj.name)


# Streamlit Page
st.set_page_config(layout="wide")
st.title('Pothole Map')
empty1, map_layout, img_layout, empty2 = st.columns([0.1,1.0,0.4,0.1])


with map_layout:
    fig = px.scatter_mapbox(lat=latitude, lon=longitude, mapbox_style="carto-positron",
                            center={'lat':37.56, 'lon':126.97}, zoom=12)
    selected_point = plotly_events(fig, click_event=True, hover_event=False)
    if st.button('Refresh'):
        selected_point = []
    
with img_layout:
    if len(selected_point) != 0:
        st.header('Image Preview')
        selected_blob = before_bucket.get_blob(df.index[selected_point[0]['pointIndex']])
        selected_img = Image.open(BytesIO(selected_blob.download_as_bytes()))
        st.image(selected_img)
        st.write(selected_blob.name)
        if st.button('보수완료'):
            tranf_blob(selected_blob, before_bucket, after_bucket)
            selected_point = []