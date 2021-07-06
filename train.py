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
import argparse
import logging
import math
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda import amp
import torchvision


from dataset import DatasetFromHdf5, Make_Train_Hdf5, Make_Test_Hdf5
from network import VDSR

parser = argparse.ArgumentParser(description="Accurate Image Super-Resolution Using Very Deep Convolutional Networks")
parser.add_argument("--dataroot", type=str, default="./data", help="Path to datasets. (default:`./data`)")
parser.add_argument("--epochs", default=100, type=int, metavar="N", help="Number of total epochs to run. (default:100)")
parser.add_argument("-b", "--batch-size", default=16, type=int, metavar="N", help="mini-batch size (default: 16)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. (default:0.1)")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, (default:0.9)")
parser.add_argument("--weight-decay", default=0.0001, type=float, help="Weight decay. (default:0.0001).")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. (default:0.4).")
parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4], help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--weights", default="", help="Path to weights (to continue training).")
parser.add_argument("-p", "--print-freq", default=5, type=int, metavar="N", help="Print frequency. (default:5)")
parser.add_argument("--manualSeed", type=int, default=0, help="Seed for initializing training. (default:0)")

args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'./training_{args.scale}x.log', filemode='w', format="%(asctime)s %(message)s", level=logging.INFO)
logger.info(f'parameter setting: {args}')

if not os.path.exists(f'./data/test_{args.scale}x.h5'):
    make_data = Make_Train_Hdf5(scale=args.scale)
    make_data('./data/train')

if not os.path.exists(f'./data/test_{args.scale}x.h5'):
    make_data = Make_Test_Hdf5(scale=args.scale)
    make_data('./data/test')

os.makedirs("./weights", exist_ok=True)
os.makedirs(f"./results/SR_{args.scale}x", exist_ok=True)

random.seed(0)
torch.manual_seed(0)

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'you are using {device} to train')

train_dataset = DatasetFromHdf5(f"{args.dataroot}/train_{args.scale}x.h5")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataset = DatasetFromHdf5(f"{args.dataroot}/test_{args.scale}x.h5")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)


model = VDSR().to(device)

if args.weights:
    model.load_state_dict(torch.load(args.weights, map_location=device))

criterion = nn.L1Loss(size_average=False).to(device)
# init lr: 0.1 decay 0.00001, so decay step 5.
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_rgb_psnr = 0.
best_y_psnr = 0.

# Creates a GradScaler once at the beginning of training.
scaler = amp.GradScaler()

for epoch in range(args.epochs):
    logger.info(f'Epoch: {epoch}, Lr: {optimizer.param_groups[0]["lr"]}')
    model.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for iteration, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # Scales loss.  Calls backward() on scaled loss to
        # create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose
        # for corresponding forward ops.
        #scaler.scale(loss).backward()

        # Adjustable Gradient Clipping.
        #nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # scaler.step() first unscales the gradients of
        # the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs,
        # optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        #scaler.step(optimizer)

        # Updates the scale for next iteration.
        #scaler.update()
        #logger.info(f"[{epoch + 1}/{args.epochs}][{iteration + 1}/{len(train_dataloader)}] "f"Loss: {loss.item():.6f} ")
        #progress_bar.set_description(f"[{epoch + 1}/{args.epochs}][{iteration + 1}/{len(train_dataloader)}] "f"Loss: {loss.item():.6f} ")

    # Test
    model.eval()
    rgb_psnr = 0.
    y_psnr = 0.
    rgb_ssim = 0.
    y_ssim = 0.
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for iteration, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            prediction = torch.clamp(model(inputs), 0.0, 1.0)
            torchvision.utils.make_grid(prediction, nrow=1)
            torchvision.utils.save_image(prediction, f"./results/SR_{args.scale}x/{iteration}.jpg")
            pred = prediction.cpu().numpy().squeeze().transpose(1, 2, 0) * 255.0
            targ = targets.cpu().numpy().squeeze().transpose(1, 2, 0) * 255.0
            rgb_psnr_ = peak_signal_noise_ratio(targ, pred, data_range=255.0)
            rgb_ssim_ = structural_similarity(targ, pred, data_range=255.0, multichannel=True)
            rgb_psnr += rgb_psnr_
            rgb_ssim += rgb_ssim_

            hr_lum = Image.fromarray(targ.astype('uint8'), 'RGB').convert("YCbCr").split()[0]
            lr_lum = Image.fromarray(pred.astype('uint8'), 'RGB').convert("YCbCr").split()[0]
            y_psnr_ = peak_signal_noise_ratio(np.array(hr_lum), np.array(lr_lum), data_range=255.0)
            y_ssim_ = structural_similarity(np.array(hr_lum), np.array(lr_lum), data_range=255.0)
            y_psnr += y_psnr_
            y_ssim += y_ssim_

            logger.info(f"Epoch: {epoch + 1} [{iteration + 1}/{len(val_dataloader)}], RGB_PSNR: {rgb_psnr_:.2f}, Y_PSNR: {y_psnr_:.2f}, RGB_SSIM: {rgb_ssim_:.2f}, Y_SSIM: {y_ssim_:.2f}.")
            #progress_bar.set_description(f"Epoch: {epoch + 1} [{iteration + 1}/{len(val_dataloader)}] "f"Loss: {mse.item():.6f} " f"PSNR: {psnr:.2f}.")

    logger.info(f"Average RGB_PSNR: {rgb_psnr / len(val_dataloader):.2f} dB.")
    logger.info(f"Average Y_PSNR: {y_psnr / len(val_dataloader):.2f} dB.")
    logger.info(f"Average RGB_SSIM: {rgb_ssim / len(val_dataloader):.2f} dB.")
    logger.info(f"Average Y_SSIM: {y_ssim / len(val_dataloader):.2f} dB.")

    # Dynamic adjustment of learning rate.
    scheduler.step()

    # Save model
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"./weights/vdsr_{args.scale}x_epoch_{epoch + 1}.pth")
    if rgb_psnr > best_rgb_psnr and y_psnr > best_y_psnr:
        best_rgb_psnr = rgb_psnr
        best_y_psnr = y_psnr
        torch.save(model.state_dict(), f"./weights/vdsr_{args.scale}x_best.pth")