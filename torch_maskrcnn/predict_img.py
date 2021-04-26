# this line use predict the one of image and show the result

import numpy as np
import cv2
import torch
import os
import glob
from show_example import showbbox
import torch
from show_example import showbbox
from train import get_model_instance_segmentation, get_transform
import transforms as T
import torchvision


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path = "im"


def eval_show(img):
    transform1 = T.Compose([
        T.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )
    num_class = 2
    model = get_model_instance_segmentation(num_class)
    model.load_state_dict(torch.load("test.pth"))
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    xx, _ = transform1(img, 0)
    xx = showbbox(model, xx)
    return xx


if __name__ == '__main__':

    pa = glob.glob(f"{path}/*.jpeg")

    for img_name in pa:
        color_image = cv2.imread(img_name)
        color_image = cv2.resize(color_image, (512, 512))

        image = eval_show(color_image)

        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', image)
        cv2.waitKey(0)

