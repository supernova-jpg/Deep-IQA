"""
The CNN Model for FR-IQA
-------------------------

KVASS Tastes good!
"""

import math
import torch
import torch.nn as nn


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=(1,1), padding=(1,1), bias=True), 
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MaxPool2x2(nn.Module):
    """
    读者注：经过一次池化运算后，图像大小减半。
    """
    def __init__(self):
        super(MaxPool2x2, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=(2,2), padding=(0,0))
    
    def forward(self, x):
        return self.pool(x)

class DoubleConv(nn.Module):
    """
    读者注：这实际上是一个步长为1，填充为1，卷积核大小为3的三维卷积层，使用的是入参和出参分别是输入/输出张量的维度。
    在经过一次卷积操作后，有公式 s2 = (W- F + 2P)/S +1 = (W - 3 + 2) +1  = W, 即经过一次卷积运算之后图像大小没有发生任何改变。
    这个doubleconv实际上是一个经过两次卷积运算和一次池化运算的layer,故每经过一个layer之后图像大小减半。
    """
    def __init__(self, in_dim, out_dim):
        super(DoubleConv, self).__init__()
        self.conv1 = Conv3x3(in_dim, out_dim)
        self.conv2 = Conv3x3(out_dim, out_dim)
        self.pool = MaxPool2x2()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.pool(y)
        return y

class SingleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleConv, self).__init__()
        self.conv = Conv3x3(in_dim, out_dim) #这儿用到了python的继承特性。
        self.pool = MaxPool2x2()

    def forward(self, x):
        y = self.conv(x)
        y = self.pool(y)
        return y


class IQANet(nn.Module):
    """
    The CNN model for full-reference image quality assessment
    
    Implements a siamese network at first and then there is regression
    """
    def __init__(self, weighted=False):
        super(IQANet, self).__init__()

        self.weighted = weighted

        # Feature extraction layers
        self.fl1 = DoubleConv(3, 32) # 图像大小：32x32x3 -> 16x16x32
        self.fl2 = DoubleConv(32, 64) # 图像大小：16x16x32 -> 8x8x64
        self.fl3 = DoubleConv(64, 128) # 图像大小：8x8x64 -> 4x4x128

        # Fusion layers
        self.cl1 = SingleConv(128*3, 128) # 图像大小：4x4x374 -> 2x2x128
        self.cl2 = nn.Conv2d(128, 64, kernel_size=2) 
        """
        这一层中，s=1, f=2, p=0 图像大小：2x2x128 -> 1x1x64
        """

        # Regression layers
        self.rl1 = nn.Linear(64, 32) # 全连接层，参数个数从64减为32
        self.rl2 = nn.Linear(32, 1)# 全连接层，参数个数从32减为1

        if self.weighted:
            self.wl1 = nn.Linear(64, 32)
            self.wl2 = nn.Linear(32, 1)

        self._initialize_weights()

    def extract_feature(self, x):
        """ Forward function for feature extraction of each branch of the siamese net """
        y = self.fl1(x)
        y = self.fl2(y)
        y = self.fl3(y)
        # 图像大小已经变成4x4x128

        return y
        
    def forward(self, x1, x2):
        """ x1 as distorted and x2 as reference """
        n_imgs, n_ptchs_per_img = x1.shape[0:2]
        # 读取原始图像和压缩图像的形状，一般这是一个四维张量。
        x1 = x1.view(-1,*x1.shape[-3:])
        """view 方法是用来改变张量的形状而不改变其数据的。
        使用了 Python 的解包操作符 *，它将 x1.shape[-3:] 返回的最后三个维度的大小作为单独的参数传递给 view 方法。
        这意味着张量 x1 被重塑为一个新形状，其最后三个维度保持不变，而第一个维度的大小根据总元素数量和其他维度的大小自动计算。
        """
        x2 = x2.view(-1,*x2.shape[-3:])

        f1 = self.extract_feature(x1)
        f2 = self.extract_feature(x2)

        f_com = torch.cat([f2, f1, f2-f1], dim=1)  # 图像大小已经变成4x4x374
        f_com = self.cl1(f_com)
        f_com = self.cl2(f_com)
        # 图像大小已经变成64

        flatten = f_com.view(f_com.shape[0], -1)

        y = self.rl1(flatten)
        y = self.rl2(y)

        if self.weighted:
            w = self.wl1(flatten)
            w = self.wl2(w)
            w = torch.nn.functional.relu(w) + 1e-8
            # Weighted average
            y_by_img = y.view(n_imgs, n_ptchs_per_img)
            w_by_img = w.view(n_imgs, n_ptchs_per_img)
            score = torch.sum(y_by_img*w_by_img, dim=1) / torch.sum(w_by_img, dim=1)
        else:
            # Calculate average score for each image
            score = torch.mean(y.view(n_imgs, n_ptchs_per_img), dim=1)

        # On the validation and the testing phase, squeeze the score to a scalar
        return score.squeeze()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            else:
                pass