# Copyright 2021 Applied Computational Intelligence Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import h5py
import cv2
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Make_Hdf5(nn.Module):
    def __init__(self, scale=4, patch_size=40, stride=None, if_aug=True):
        super(Make_Hdf5, self).__init__()
        self.scale =scale
        self.patch_size = patch_size
        if (stride == None):
            self.stride = patch_size
        self.if_aug = if_aug

    def read_data(self, img):
        raw_img = Image.open(img).convert("RGB")
        w, h = raw_img.size    
        w =  w - (w % self.scale)
        h =  h - (h % self.scale)
        hr = raw_img.crop((0, 0, w, h))
        temp_lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        lr = temp_lr.resize((w, h), Image.BICUBIC)
        return hr, lr
        
    def augment(self, hr, lr):
        rand = np.random.rand()
        if rand > 0.5:
            hr = cv2.flip(hr, 0)
            lr = cv2.flip(lr, 0)
        rand = np.random.rand()
        if rand > 0.5:
            hr = cv2.flip(hr, 1)
            lr = cv2.flip(lr, 1)
        return hr, lr

    def make_train(self, data_path):
        train_path = os.path.join(data_path, 'train')
        image_filenames = [os.path.join(train_path, x) for x in os.listdir(train_path)]
        sub_hr = []
        sub_lr = []
        for img in image_filenames:
            hr, lr = self.read_data(img)
            for i in range(0, hr.size[0] - self.patch_size + 1, self.stride):
                for j in range(0, hr.size[1] - self.patch_size + 1, self.stride):
                    hr_patch = hr.crop((i, j, i + self.patch_size, j + self.patch_size))
                    lr_patch = lr.crop((i, j, i + self.patch_size, j + self.patch_size))
                    sub_hr.append(np.array(hr_patch))
                    sub_lr.append(np.array(lr_patch))
        sub_hr = np.array(sub_hr)
        sub_lr = np.array(sub_lr)
        print('input shape : ',sub_lr.shape)
        print('label shape : ',sub_hr.shape)
        with h5py.File(f'./data/train_{self.scale}x.h5', 'w') as hf:
            hf.create_dataset('input', data=sub_lr)
            hf.create_dataset('target', data=sub_hr)

    def make_test(self, data_path):
        test_path = os.path.join(data_path, 'test')
        image_filenames = [os.path.join(test_path, x) for x in os.listdir(test_path)]
        sub_hr = []
        sub_lr = []
        for img in image_filenames:
            hr, lr = self.read_data(img)
            sub_hr.append(np.array(hr))
            sub_lr.append(np.array(lr))
        sub_hr = np.array(sub_hr)
        sub_lr = np.array(sub_lr)
        print('input shape : ',sub_lr.shape)
        print('label shape : ',sub_hr.shape)
        with h5py.File(f'./data/test_{self.scale}x.h5', 'w') as hf:
            hf.create_dataset('input', data=sub_lr)
            hf.create_dataset('target', data=sub_hr)

    def forward(self, data_path):
        self.make_train(data_path)
        self.make_test(data_path)  


class DatasetFromHdf5(Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path, 'r') 
        self.input = hf.get('input')
        self.target = hf.get('target')

    def __getitem__(self, index):
        lr = torch.from_numpy(self.input[index,:,:,:].transpose(2, 0, 1)  / 255.0 ).float()
        hr = torch.from_numpy(self.target[index,:,:,:].transpose(2, 0, 1) / 255.0).float()
        return lr, hr
        
    def __len__(self):
        return self.input.shape[0]

if __name__ == '__main__':
    make_data = Make_Hdf5()
    make_data('./data')
