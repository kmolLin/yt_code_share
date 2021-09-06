
import os
import numpy as np
import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
from train import PennFudanDataset, get_transform, get_model_instance_segmentation
import utils
import cv2


def showbbox(model, img):
    # 輸入的img是0-1範圍的tensor
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    with torch.no_grad():
        '''
        prediction形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])

    # print(prediction)

    img = img.permute(1, 2, 0)  # C,H,W → H,W,C，用來畫圖
    img = (img * 255).byte().data.cpu()  # * 255，float轉0-255
    img = np.array(img)  # tensor → ndarray
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = np.array(prediction[0]['masks'].detach().cpu() * 255)
    mask = mask.astype("uint8")

    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())
        label = prediction[0]['labels'][i].item()
        mm = mask[i][0]
        contours, hierarchy = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Draw contours:
        if label == 1:
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
            cv2.putText(img, 'mark_type_1', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
        elif label == 2:
            cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
            cv2.putText(img, 'mark_type_2', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

    # plt.figure(figsize=(20, 15))
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


if __name__ == "__main__":
    num_class = 2
    model = get_model_instance_segmentation(num_class)
    model.load_state_dict(torch.load("test.pth"))
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    img, _ = dataset_test[1]
    showbbox(model, img)

