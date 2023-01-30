import sys
sys.path.append('/opt/ml/final-project-level3-cv-03')

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response
import mmcv
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from mmdetection.mmdet.apis import init_detector, inference_detector

import io
import os
import json
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any

from google.cloud import storage
from google.cloud import bigquery

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/opt/ml/essential_config/aitech4-finalproject-0000-b7e0b3c932a1.json'

storage_client = storage.Client()
bigquery_client = bigquery.Client()

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
threshold = 0.7
## 주요 수정사항 ##



@router.post('/OD', description='Object Detection _ Where Pothole is')
async def load_image(files : UploadFile = File(...)):
    img = Image.open(BytesIO(await files.read()))
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
        # coord 임시값
        coord = {'x' : 38.0, 'y' : 128.0}
        meta_id = str(uuid4())
        meta_date = datetime.today().strftime("%Y%m%d%H%M%S")
        
        byted_file = io.BytesIO()
        img.save(byted_file, format=img.format)
        source_file = io.BytesIO(byted_file.getvalue())
        upload_filename = f'{meta_date}_{meta_id[:8]}.jpg'
        
        bbox_json = json.dumps(bbox_record)
        metadata = {
            'uuid' : meta_id,
            'datetime' : meta_date,
            'bbox' : bbox_json,
            'lat' : coord['x'],
            'lon' : coord['y']
        }
        bigquery_client.insert_rows(table_id, [metadata], schema)
        
        bucket = storage_client.bucket('detected_potholes')
        blob = bucket.blob(upload_filename)
        blob.metadata = metadata

        blob.upload_from_file(source_file)
        print('upload completed')
        
    print(result)
    return Response()