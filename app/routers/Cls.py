from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.param_functions import Depends
from routers.model import get_model, get_config, predict_from_image_byte

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


## 모델 체크포인트 경로 ##
my_model = get_model("/opt/ml/essential_config/best_classification.pth")
router = APIRouter()


# Inference & Logging to Bigquery & Load image to GCS
@router.post("/IC", description="Image Classification _ Pothole or not")
async def predict_image(files: List[UploadFile] = File(...),
                        config: Dict[str, Any] = Depends(get_config)):
    image_bytes = await files[0].read()
    result = predict_from_image_byte(model=my_model, image_bytes=image_bytes, config=config)
    
    # 임시값 _ request에서 받을 예정
    coord = {'x' : 38.0, 'y' : 128.0}
    
    if result[0] == 'pothole':
        meta_id = str(uuid4())
        meta_date = datetime.today().strftime("%Y%m%d%H%M%S")

        source_file = io.BytesIO(image_bytes)
        upload_filename = f'{meta_date}_{meta_id[:8]}.jpg'
        
        # 임시값 _ detection 적용 예정
        bbox_info = {
            'b1' : [1, 2, 3, 4, 5],
            'b2' : [6, 7, 8, 9, 0]
        }
        bbox_json = json.dumps(bbox_info)
        
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
    return JSONResponse(content = {'result' : result[0]})

