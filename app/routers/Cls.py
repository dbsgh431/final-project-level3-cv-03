from fastapi import APIRouter, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Optional, Dict, Any
from routers.model import get_model, get_config, predict_from_image_byte

## 모델 체크포인트 경로 ##
my_model = get_model("/opt/ml/final-project-level3-cv-03/classification/assets/best.pth")



router = APIRouter()

class InferenceResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    result: Optional[List]
    
@router.post("/IC", description="Image Classification _ Pothole or not")
async def predict_image(files: List[UploadFile] = File(...),
                config: Dict[str, Any] = Depends(get_config)):
    image_bytes = await files[0].read()
    result = predict_from_image_byte(model=my_model, image_bytes=image_bytes, config=config)
    result = InferenceResult(result=result)
    return result

