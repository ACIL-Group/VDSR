import cv2
import os
import math
import numpy as np
from skimage.color import rgb2ycbcr
from skimage.measure import compare_psnr

def Bicubic_SR(data_path, scale):
    os.makedirs(f'./results/bicubic_{scale}x', exist_ok=True)
    rgb_psnr = []
    y_psnr = []
    for img in os.listdir(data_path):
        image_filenames = os.path.join(data_path, img)
        print(image_filenames)
        raw_img = cv2.imread(image_filenames)
        h, w = raw_img.shape[0], raw_img.shape[1] # cv2 return rows and column numbers. width: im.shape[1], height: im.shape[0]
        h =  h - (h % scale)
        w =  w - (w % scale)
        hr = raw_img[0:h, 0:w, :]
        temp_lr = cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(temp_lr, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'./results/bicubic_{scale}x/{img}', lr)
        print(compare_psnr(hr, lr, 255.0))
        rgb_psnr.append(compare_psnr(hr, lr, 255.0))

        hr_lum = rgb2ycbcr(hr)[:,:,0]
        temp_lr_lum = cv2.resize(hr_lum, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        lr_lum = cv2.resize(temp_lr_lum, (w, h), interpolation=cv2.INTER_CUBIC)
        print(compare_psnr(hr_lum, lr_lum, 255.0))
        y_psnr.append(compare_psnr(hr_lum, lr_lum, 255.0))
        
    print(f'Average RGB channel PSNR: {np.mean(rgb_psnr)}')
    print(f'Average Y channel PSNR: {np.mean(y_psnr)}')

if __name__ == '__main__':
    Bicubic_SR('./data/test', 4)