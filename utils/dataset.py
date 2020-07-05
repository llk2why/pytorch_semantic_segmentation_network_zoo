import os
import glob
import torch
import numpy as np

from PIL import Image
from os import listdir
from os.path import join,split,splitext
from utils import is_image_file
from torch.utils.data import Dataset

class TrainingDataset(Dataset):
    def __init__(self,img_dir,mask_dir,input_transform,target_transform,logger=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        mask_file_names = [file_name for file_name in listdir(mask_dir) if not file_name.startswith('.')]
        name2filename = {splitext(file_name)[0]:file_name for file_name in mask_file_names}

        self.img_file_names = [file_name for file_name in listdir(img_dir) if is_image_file(file_name)]
        self.mask_file_names = [name2filename[splitext(file_name)[0]] for file_name in self.img_file_names]

        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self,i):
        img_path = join(self.img_dir,self.img_file_names[i])
        mask_path = join(self.mask_dir,self.mask_file_names[i])

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img = np.array(img)
        mask = np.array(mask)

        if len(img.shape)==2 or img.shape[2]==1:
            zeros = np.zeros_like(img)
            # img = np.stack([img,zeros,zeros],axis=2) # if you just need one channel, then enable it
            img = np.stack([img,img,img],axis=2)
            

        assert len(mask.shape)==2,'invalid mask dimension'
        assert len(img.shape)==3 and img.shape[2]==3,'invalid input dimension'

        img = self.input_transform(img)
        mask = self.target_transform(mask)
        return img,mask

class TestingDataset(Dataset):
    def __init__(self,img_dir,input_transform,logger=None):
        self.img_dir = img_dir
        self.img_file_names = [file_name for file_name in listdir(img_dir) if is_image_file(file_name)]
        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        self.input_transform = input_transform

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self,i):
        img_path = join(self.img_dir,self.img_file_names[i])
        img = Image.open(img_path)
        img = np.array(img)

        if len(img.shape)==2 or img.shape[2]==1:
            zeros = np.zeros_like(img)
            # img = np.stack([img,zeros,zeros],axis=2) # if you just need one channel, then enable it
            img = np.stack([img,img,img],axis=2)

        assert len(img.shape)==3 and img.shape[2]==3,'invalid input dimension'

        img = self.input_transform(img)
        return img,self.img_file_names[i]

if __name__ == "__main__":
    from transform import ToLabel
    from torchvision.transforms import Compose, CenterCrop, Normalize
    from torchvision.transforms import ToTensor, ToPILImage
    input_transform = Compose([
        ToTensor(),
    ])
    target_transform = Compose([
        ToLabel(),
    ])
    dataset = TrainingDataset('/home/llv/Pytorch-SegNet/data/imgs','/home/llv/Pytorch-SegNet/data/masks',input_transform,target_transform)
    img,mask = dataset.__getitem__(0)
    print(img.shape)
    print(mask.shape)
