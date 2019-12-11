import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import transforms as ext_transforms
from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU
#from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils

from argparse import ArgumentParser



def get_arguments():
    parser = ArgumentParser()

    # model path
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default='./save/ENet_Cityscapes/120Eproch',
        help="The model path. Default: ./save/ENet_Cityscapes/120Eproch")
    # Image path
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default='',
        help="The image path. Default: ")
    return parser.parse_args()


def load_model(model_path):
    print(args.model_path)
    
    pass

def prediction(model,image_path):
    pass



if __name__ == '__main__':
    args = get_arguments()
    model = load_model(args.model_path)
    prediction(model,args.image_path)