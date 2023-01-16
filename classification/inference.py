from torch.utils.data import Dataset, DataLoader
import torch
import cv2

from model import CFG, BaseModel, Augmentation

from matplotlib import pyplot as plt


device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_img_paths = "테스트 이미지 경로"

class CustomTestDataset(Dataset):
    def __init__(self, img_paths, transforms=None):
        self.img_paths = img_paths
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        return image
    
    def __len__(self):
        return len(self.img_paths)



test_dataset = CustomTestDataset(test_img_paths, None, Augmentation.test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
model = BaseModel()


def inference(model, img_path, device):
    model.load_state_dict(torch.load("/opt/ml/model_save_dir/best.pt", map_location=device))
    model = model.to(device)
    
    model_preds = []
    img_paths = []
    img_paths.append(img_path)
    
    model.eval()
    with torch.no_grad():
        for img in iter(test_loader):
            img = img.float().to(device)
            model_pred = model(img)
            model_preds.append(torch.round(model_pred).detach().cpu().numpy())
    
    print('Done.')
    return model_preds[0][0]


preds = inference(model, test_img_paths, device)
print(preds)