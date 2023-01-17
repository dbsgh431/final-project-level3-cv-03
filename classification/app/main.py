from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Optional, Dict, Any
from app.model import MyModel, get_model, get_config, predict_from_image_byte

app = FastAPI()

class InferenceResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    result: Optional[List]
    
@app.post("/predict", description="이미지를 분류합니다.")
async def predict_image(files: List[UploadFile] = File(...),
                  model: MyModel = Depends(get_model),
                  config: Dict[str, Any] = Depends(get_config)):
    
    image_bytes = await files[0].read()
    result = predict_from_image_byte(model=model, image_bytes=image_bytes, config=config)
    result = InferenceResult(result=result)
    return result

