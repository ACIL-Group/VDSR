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
import torch
import torch.nn as nn


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        # Residual layer
        conv_relu_block = []
        for i in range(18):
            conv_relu_block.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
            conv_relu_block.append(nn.ReLU(inplace=True))
        self.conv_relu_block = nn.Sequential(*conv_relu_block)

        # last layer
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

        # Init weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.conv_relu_block(out)
        out = self.conv2(out)
        out = torch.add(out, residual)
        return out