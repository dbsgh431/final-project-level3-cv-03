import sys
sys.path.append('/opt/ml/final-project-level3-cv-03')

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import Response
from mmdetection.mmdet.apis import init_detector, inference_detector

import io
import os
import json
import numpy as np
from uuid import uuid4
from pytz import timezone
from datetime import datetime
from PIL import Image, ImageDraw

from google.cloud import storage
from google.cloud import bigquery

# GCP 사용을 위한 서비스 계정 키 _ 환경변수에 키 경로를 설정
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/opt/ml/essential_config/aitech4-finalproject-0000-b7e0b3c932a1.json'

storage_client = storage.Client()
bigquery_client = bigquery.Client()

# BigQuery의 계정.데이터셋.테이블
# 테이블 스키마
table_id = 'aitech4-finalproject-0000.pothole_serving_log.pothole_detection'
schema = [
        bigquery.SchemaField('uuid', 'STRING'),
        bigquery.SchemaField('datetime', 'INTEGER'),
        bigquery.SchemaField('bbox', 'JSON'),
        bigquery.SchemaField('lat', 'FLOAT'),
        bigquery.SchemaField('lon', 'FLOAT')
    ]

## 주요 수정사항 ##
cfg = '/opt/ml/final-project-level3-cv-03/mmdetection/configs/_custom_/cascade_swinlarge.py'
ckpt = '/opt/ml/essential_config/best_detection.pth'
model = init_detector(cfg , ckpt, device='cuda:0')
router = APIRouter()
threshold = 0.8
## 주요 수정사항 ##

# Detection & DB & Log
@router.post('/OD', description='Object Detection _ Where Pothole is')
async def load_image(files : UploadFile = File(...),
                     lat : float = Form(...),
                     lon : float = Form(...)):
    """
    files : image data (bytes file after files.read())
    lat : latitude of image
    lon : longitude of image
    """
    
    img = Image.open(io.BytesIO(await files.read()))
    result = inference_detector(model, np.array(img))[0]

    draw = ImageDraw.Draw(img)
    num_pot = 0
    bbox_record = {}
    
    for i in result:
        if len(i) != 0:
            if i[4] > threshold:
                num_pot += 1
                draw.rectangle(i[:4], outline=(255, 0, 0), width=3)
                bbox_record[f'b{num_pot}'] = i.tolist()
    
    if num_pot != 0:
        meta_id = str(uuid4())
        meta_date = datetime.now(timezone('Asia/Seoul')).strftime("%Y%m%d%H%M%S%f")

        byted_file = io.BytesIO()
        img.save(byted_file, format=img.format)
        source_file = io.BytesIO(byted_file.getvalue())
        upload_filename = f'{meta_date}_{meta_id[:8]}.jpg'
        
        bbox_json = json.dumps(bbox_record)
        metadata = {
            'uuid' : meta_id,
            'datetime' : meta_date,
            'bbox' : bbox_json,
            'lat' : lat,
            'lon' : lon
        }
        bigquery_client.insert_rows(table_id, [metadata], schema)
        
        bucket = storage_client.bucket('detected_potholes')
        blob = bucket.blob(upload_filename)
        blob.metadata = metadata

        blob.upload_from_file(source_file)

    return Response()