from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# 想要的輸出圖像尺寸
imsize = 512 if use_cuda else 128  # 如果沒有 GPU 則使用小尺寸

loader = transforms.Compose([
    transforms.Scale((imsize, 512)),  # 縮放圖像
    transforms.ToTensor()])  # 將其轉化為 torch 張量

def image_loader(image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = Variable(loader(image))
    # 由於身經網路輸入的需要, 增加 batch 維度
    image = image.unsqueeze(0)
    return image

style_img = image_loader("style11.jpg").type(dtype)
content_img = image_loader("render2_backup.png").type(dtype)
print(content_img.size())

unloader = transforms.ToPILImage()  # 轉回 PIL 圖片

plt.ion()

def imshow(tensor, title=None):
    image = tensor.clone().cpu()  # 複製他是為了不改變
    image = image.view(3, imsize, imsize)  # 移除 batch 维度
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 暂停一会, 讓圖片更新

plt.figure()
imshow(style_img.data, title='Style Image')

plt.figure()
imshow(content_img.data, title='Content Image')

# assert style_img.size() == content_img.size(), \
#     "we need to import style and content images of the same size"

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        # 動態機算梯度: 它是狀態值, 而不是變量.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b= 特征映射的数量
        # (c,d)= 一个特征映射的维度 (N=c*d)

        features = input.view(a * b, c * d)  # 将 F_XL 转换为 \hat F_XL

        G = torch.mm(features, features.t())

        # 標準化矩陣 (gram matrix) 的值
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

cnn = models.vgg16(pretrained=True).features

# 將模型轉移到到 GPU 上:
if use_cuda:
    cnn = cnn.cuda()

# 計算風格/内容損失的層數 :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # 僅為了有一個可叠代的列表 內容/風格 損失
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # 新建的 Sequential 網絡模組
    gram = GramMatrix()  # 我们需要一个gram module (gram module) 來計算風格

    # 可能的話將這些模塊移到 GPU 上:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # 加內容損失:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # 加風格損失:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # 加內容損失:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # 加風格損失:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses

input_img = content_img.clone()
# 如果你想用白噪聲做輸入, 請取消下面的注釋行:
# input_img = Variable(torch.randn(content_img.data.size())).type(dtype)

# 在繪圖中加入原始的輸入圖像:
plt.figure()
imshow(input_img.data, title='Input Image')

def get_input_param_optimizer(input_img):
    # 這行顯示了輸入是一個需要梯度計算的參數
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=601,
                       style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # 校正更新後的輸入圖像值
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data, content_score.data))
                print()
            if run[0] % 100 == 0:
                input_param.data.clamp_(0, 1)
                plt.figure()
                imshow(input_param.data, title=f'training step {run[0]}')

            return style_score + content_score

        optimizer.step(closure)

    # 最後一次的校正...
    input_param.data.clamp_(0, 1)

    return input_param.data

output = run_style_transfer(cnn, content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()