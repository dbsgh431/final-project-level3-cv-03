import sys
sys.path.append('/opt/ml/final-project-level3-cv-03')

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import mmcv
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from mmdetection.mmdet.apis import init_detector, inference_detector

## 주요 수정사항 ##
cfg = '/opt/ml/level3_productserving-level3-cv-03/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
ckpt = '/opt/ml/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # checkpoint in mmdetection/demo/inference_demo.ipynb
temp_img_path = '/opt/ml/level3_productserving-level3-cv-03/app/temp_image/temp.png'
model = init_detector(cfg , ckpt, device='cuda:0')
threshold = 0.6
## 주요 수정사항 ##

router = APIRouter()

@router.post('/OD', description='Object Detection _ Where Pothole is')
async def load_image(files : UploadFile = File(...)):
    img = Image.open(BytesIO(await files.read()))
    result = inference_detector(model, np.array(img))
        
    draw = ImageDraw.Draw(img)
    for i in result:
        if len(i) != 0:
            if i[0][4] > threshold:
                draw.rectangle(i[0], outline=(255, 0, 0), width=3)
            
    mmcv.imwrite(np.flip(np.array(img), axis=2), temp_img_path)
    # print(result)
    return FileResponse(temp_img_path)