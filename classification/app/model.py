import timm
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import io
import numpy as np
from typing import List, Dict, Any

global CFG

CFG = {
    'IMG_WIDTH':1280,
    'IMG_HEIGTH':720,
    # 'EPOCHS':10,
    # 'LEARNING_RATE':3e-4,
    # 'BATCH_SIZE':32,
    # 'SEED':3
}

class MyModel(nn.Module):
    def __init__(self, num_classes=1):
        super(MyModel, self).__init__()
        self.backbone = timm.create_model(model_name='efficientnet_b0', pretrained=True)
        self.fc = nn.Linear(1000,num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = nn.Sigmoid()(x)
        return x
    
def get_model(model_path: str="assets/best.pth") -> MyModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def _transform_image(image_bytes: bytes):
    transform = A.Compose([
        A.Resize(CFG['IMG_HEIGTH'],CFG['IMG_WIDTH']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        A.ToGray(p=1),
        ToTensorV2()
        ])
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image_array = np.array(image)
    return transform(image=image_array)["image"].unsqueeze(0)

def predict_from_image_byte(model: MyModel, image_bytes: bytes, config: Dict[str, Any]) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformed_image = _transform_image(image_bytes).to(device)
    output = model.forward(transformed_image)
    return config["classes"][int(torch.round(output[0]))]


def get_config(config_path: str = "assets/config.yaml"):
    import yaml

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
