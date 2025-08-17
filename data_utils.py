import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(321+i)  # Changed from 211+i to 321+i  
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'clear')
        print(f"Found {len(self.haze_imgs)} hazy images in {path}/input")
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,20000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split('_')[0]
        clear_name=id+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)
    
class RW2AH_Dataset(data.Dataset):
    def __init__(self, path, train=True, size='whole_img', format='.png'):
        super(RW2AH_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        
        # Load directories
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'input'))
        self.haze_imgs = [os.path.join(path, 'input', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'gt')
         # Only check mask directory during training
        if self.train:
            self.mask_dir = os.path.join(path, 'mask')  # Mask directory
            assert os.path.exists(self.mask_dir), f"Mask directory {self.mask_dir} does not exist"
        print(f"Found {len(self.haze_imgs)} hazy images in {path}/input")
    
    def __getitem__(self, index):
        # Load hazy and clear images
        haze = Image.open(self.haze_imgs[index]).convert('RGB')
        img_name = self.haze_imgs_dir[index]  # E.g., '1-0001-03.png'
        clear = Image.open(os.path.join(self.clear_dir, img_name)).convert('RGB')
        
        # Load masks
        if self.train:
            weight_mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '_weight.png'))
            binary_mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '_binary.png'))
            try:
                weight_mask = Image.open(weight_mask_path).convert('L')  # Single channel
                binary_mask = Image.open(binary_mask_path).convert('L')  # Single channel
            except Exception as e:
                print(f"Error loading masks for {img_name}: {e}")
                raise e
        
        # Resizing or cropping
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = tfs.functional.crop(haze, i, j, h, w)
            clear = tfs.functional.crop(clear, i, j, h, w)
            if self.train:
                weight_mask = tfs.functional.crop(weight_mask, i, j, h, w)
                binary_mask = tfs.functional.crop(binary_mask, i, j, h, w)
        
        # Apply augmentations
        if self.train:
            haze, clear, weight_mask, binary_mask = self.augData(haze, clear, weight_mask, binary_mask)
            return haze, clear, weight_mask, binary_mask
        else:
            haze, clear = self.augTest(haze, clear)
            return haze, clear
    
    def augData(self, data, target, weight_mask, binary_mask):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            # Horizontal flip
            if rand_hor:
                data = tfs.RandomHorizontalFlip(1)(data)
                target = tfs.RandomHorizontalFlip(1)(target)
                weight_mask = tfs.RandomHorizontalFlip(1)(weight_mask)
                binary_mask = tfs.RandomHorizontalFlip(1)(binary_mask)
            # Rotation
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
                weight_mask = FF.rotate(weight_mask, 90 * rand_rot)
                binary_mask = FF.rotate(binary_mask, 90 * rand_rot)
        
        # Transform images
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        
        # Transform masks
        weight_mask = tfs.ToTensor()(weight_mask) * 1.5  # [170, 255] -> [0.667, 1.0] -> [1.0, 1.5]
        binary_mask = tfs.ToTensor()(binary_mask)  # uint8 [0, 255] -> [0, 1]
        
        return data, target, weight_mask, binary_mask
    def augTest(self,data,target):
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target

    
    def __len__(self):
        return len(self.haze_imgs)
    
'''class RW2AH_Dataset(data.Dataset):
    def __init__(self, path, train=True, size='whole_img', format='.png'):
        super(RW2AH_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        
        # Load the haze and clear images
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'input'))
        self.haze_imgs = [os.path.join(path, 'input', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'gt')
        self.mask_dir = os.path.join(path, 'mask')  # Mask directory
        assert os.path.exists(self.mask_dir), f"Mask directory {self.mask_dir} does not exist"
        print(f"Found {len(self.haze_imgs)} hazy images in {path}/input")
        
    def __getitem__(self, index):
        # Load the haze image
        haze = Image.open(self.haze_imgs[index])
        
        # Generate the corresponding clear image filename
        img_name = self.haze_imgs_dir[index]  # E.g., '1-0001-03.png'
        clear_name = img_name  # Directly map the file name
        clear = Image.open(os.path.join(self.clear_dir, clear_name))

        # Resizing or cropping operations
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(
                haze, output_size=(self.size, self.size)
            )
            haze = tfs.functional.crop(haze, i, j, h, w)
            clear = tfs.functional.crop(clear, i, j, h, w)
        
        # Apply augmentations
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear
    
    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot=random.randint(0,3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target
    
    def __len__(self):
        return len(self.haze_imgs)'''

import os
pwd=os.getcwd()
print(pwd)
path='/home/klay/papersToReproduce/datasets/RW2AH'#path to your 'data' folder

RW2AH_train_loader=DataLoader(dataset=RW2AH_Dataset(path+'/train',train=True,size=crop_size),batch_size=BS,shuffle=True)
RW2AH_test_loader=DataLoader(dataset=RW2AH_Dataset(path+'/test',train=False,size='whole img'),batch_size=1,shuffle=False)

#ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/ITS',train=True,size=crop_size),batch_size=BS,shuffle=True)
#ITS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'//RESIDE/SOTS/indoor',train=False,size='whole img'),batch_size=1,shuffle=False)

#OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/OTS',train=True,format='.jpg'),batch_size=BS,shuffle=True)
#OTS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/outdoor',train=False,size='whole img',format='.png'),batch_size=1,shuffle=False)

if __name__ == "__main__":
    pass
