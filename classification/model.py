import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout

import time

from PIL import Image
from tqdm.auto import tqdm

import random
import numpy as np
import os
import cv2

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import timm

import pandas as pd


import wandb

global CFG

CFG = {
    'IMG_WIDTH':1280,
    'IMG_HEIGTH':720,
    'EPOCHS':10,
    'LEARNING_RATE':0.0001,
    'BATCH_SIZE':16,
    'SEED':3
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def crop_three_quarters(img, h):
    h_start = int(h*0.25)
    image = img[h_start:h+1,::,::]
    return image

class Augmentation():
    train_transform = A.Compose([A.Resize(CFG['IMG_HEIGTH'],CFG['IMG_WIDTH']),
                                A.HorizontalFlip(p=0.5),
                                A.RandomBrightnessContrast(p=0.25),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                A.ToGray(p=1),
                                ToTensorV2()
                                ])
    val_transform = A.Compose([
                                A.Resize(CFG['IMG_HEIGTH'],CFG['IMG_WIDTH']),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                A.ToGray(p=1),
                                ToTensorV2()
                                ])
    test_transform = A.Compose([
                                A.Resize(CFG['IMG_HEIGTH'],CFG['IMG_WIDTH']),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                A.ToGray(p=1),
                                ToTensorV2()
                                ])


class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths['filepath']
        self.labels = labels['info']
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = crop_three_quarters(image, CFG['IMG_HEIGTH'])
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
            

        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image
    
    def __len__(self):
        return len(self.img_paths)


class BaseModel(nn.Module):
    def __init__(self, num_classes=1):
        super(BaseModel, self).__init__()
        self.backbone = timm.create_model(model_name='efficientnet_b0', pretrained=True)
        self.fc = nn.Linear(1000,num_classes)
        #self.classifier1 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = nn.Sigmoid()(x)
        return x

def calculate_accuracy(y_pred, y):
    top_pred = torch.round(y_pred)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def validation(model, criterion, test_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    val_acc = []
    
    with torch.no_grad():
        for img, label in iter(test_loader):
            img, label = img.float().to(device), label.float().to(device).reshape(-1,1)
            
            model_pred = model(img)

            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())

            val_acc.append(calculate_accuracy(model_pred, label).item())
            
            true_labels += label.detach().cpu().numpy().tolist()

    return np.mean(val_loss), np.mean(val_acc)

def time_of_epoch(start, end):
    time_per_epoch = end - start
    time_per_epoch_min = int(time_per_epoch / 60)
    time_per_epoch_sec = int(time_per_epoch -(time_per_epoch_min*60))
    return time_per_epoch_min, time_per_epoch_sec


def train(model, optimizer, train_loader, test_loader, scheduler, device):
    model.to(device)

    criterion = nn.BCELoss().to(device)

    best_score = 0
    best_model = None

    for epoch in range(1, CFG["EPOCHS"]+1):
        model.train()
        start_time = time.monotonic()
        train_loss = []
        epoch_acc = []
        for step,(img, label) in enumerate(train_loader): #tqdm(iter(train_loader)

            img, label = img.float().to(device), label.float().to(device)
            optimizer.zero_grad()
            label = label.view(-1,1)

            model_pred = model(img)

            acc = calculate_accuracy(model_pred, label)
            epoch_acc.append(acc.item())

            loss = criterion(model_pred, label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if (step + 1) % 5 == 0:
                print(f'Epoch [{epoch}], Step [{step+1}], Train Loss : [{round(loss.item(),4):.5f}]')
                wandb.log({'Train Loss':round(loss.item(),4)})
                
        epoch_acc = np.mean(epoch_acc)
        wandb.log({'train_acc':epoch_acc})
        end_time = time.monotonic()
        epoch_min, epoch_sec = time_of_epoch(start_time, end_time)
        train_loss_m = np.mean(train_loss)
        val_loss, val_acc = validation(model, criterion, test_loader, device)

        print(f'Epoch [{epoch}], Train Loss : [{train_loss_m:.5f}] Val Loss : [{val_loss:.5f}] Val acc : [{val_acc:.5f}],Time : {epoch_min}m {epoch_sec}s')
        wandb.log({'val_acc':val_acc})
        if scheduler is not None:
            scheduler.step()
            
        if best_score < val_acc:
            best_model = model
            best_score = val_acc
            print(f"save_best_pth EPOCH {epoch}")
            torch.save(model.state_dict(),"/opt/ml/model_save_dir/best_efb0_wbtest.pth")
    wandb.save("/opt/ml/model_save_dir/best_efb0_wbtest.pth")
    return best_model

if __name__ == '__main__':

    wandb.init(
    project="Final Project", 
    entity="aitech4_cv3",
    name='classification_EFB0_wandb_test',
    config = {
        "lr" : CFG['LEARNING_RATE'],
        "epoch" : CFG['EPOCHS'],
        "batch_size" : CFG['BATCH_SIZE'],
    }
    )
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_file = pd.read_csv('/opt/ml/data_csv/data.csv')
    label = data_file['info']
    data =  data_file.drop(['info',"Unnamed: 0"], axis=1)
    X_train, y_train, X_test, y_test = train_test_split(data, label, test_size=0.2, random_state=CFG['SEED'])


    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    y_train = y_train.reset_index()
    y_test = y_test.reset_index()

    seed_everything(CFG['SEED'])

    train_dataset = CustomDataset(X_train, X_test, Augmentation.train_transform)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(y_train, y_test, Augmentation.val_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = BaseModel()
    optimizer = optim.AdamW(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) 
    infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)
