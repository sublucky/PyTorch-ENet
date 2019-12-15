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

import  copy
import matplotlib.pyplot as plt
import collections   
from visdom import Visdom

COLOR_ECODING_DICT = collections.OrderedDict() 

COLOR_ECODING_DICT['unlabeled']     = (0, 0, 0)
COLOR_ECODING_DICT['road']          = (128, 64, 128)
COLOR_ECODING_DICT['sidewalk']      = (244, 35, 232)
COLOR_ECODING_DICT['building']      = (70, 70, 70)
COLOR_ECODING_DICT['wall']          = (102, 102, 156)
COLOR_ECODING_DICT['fence']         = (190, 153, 153)
COLOR_ECODING_DICT['pole']          = (153, 153, 153)
COLOR_ECODING_DICT['traffic_light'] = (250, 170, 30)
COLOR_ECODING_DICT['traffic_sign']  =  (220, 220, 0)
COLOR_ECODING_DICT['vegetation']    = (107, 142, 35)
COLOR_ECODING_DICT['terrain']       = (152, 251, 152)
COLOR_ECODING_DICT['sky']           = (70, 130, 180)
COLOR_ECODING_DICT['person']        = (220, 20, 60)
COLOR_ECODING_DICT['rider']         = (255, 0, 0)
COLOR_ECODING_DICT['car']           = (0, 0, 142)
COLOR_ECODING_DICT['truck']         = (0, 0, 70)
COLOR_ECODING_DICT['bus']           = (0, 60, 100)
COLOR_ECODING_DICT['train']         = (0, 80, 100)
COLOR_ECODING_DICT['motorcycle']    = (0, 0, 230)
COLOR_ECODING_DICT['bicycle']       = (119, 11, 32)


def get_arguments():
    parser = ArgumentParser()

    # model path
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default='./save/epoch100',
        help="The model path. Default: ./save/epoch100")
    # Image path
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default='./bonn_000040_000019_leftImg8bit.png',
        help="The image path. Default: ")
    return parser.parse_args()

def create_model(num_classes=20,device='cuda'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ENet(num_classes).to(device)
    return model

def load_checkpoint(model,folder_dir,filename='ENet'):
    print(folder_dir)

    assert os.path.isdir(
    folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    #before_state = copy.deepcopy(model.state_dict()) 
    model.load_state_dict(checkpoint['state_dict'])
    #after_state = copy.deepcopy(model.state_dict()) 
    #model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def prediction(model,image_path):
    # Convert image to tensor
    img = Image.open(image_path)#返回一个Image对象
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create transform
    image_transform = transforms.Compose(
                                        [transforms.Resize((480, 360)),
                                        transforms.ToTensor()])
    img_tensor = image_transform(img)
    img_tensor.unsqueeze_(0)
    img_tensor_gpu = img_tensor.to(device)

    # predict
    predictions_gpu = model(img_tensor_gpu)
    
    # convert label_tensor to image
    unloader = transforms.ToPILImage()

    # convert predict to pil image
    _, label_tensor_gpu = torch.max(predictions_gpu.data, 1)
    label_tensor_cpu = label_tensor_gpu.cpu().clone()

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(COLOR_ECODING_DICT),
        transforms.ToTensor()
    ])
    
    color_tensor_cpu = label_to_rgb(label_tensor_cpu)
    color_img = unloader(color_tensor_cpu)
    print(label_tensor_cpu)

    print(color_img)

    img_tensor_cpu = img_tensor_gpu.cpu().clone()
    img_tensor_cpu = img_tensor_cpu.squeeze(0)
    img_img = unloader(img_tensor_cpu)

    plt.figure()
    plt.imshow(img_img)
    plt.figure()
    plt.imshow(color_img)
    plt.show()
    pass



if __name__ == '__main__':
    args = get_arguments()
    model = create_model(num_classes=20,device='cuda')
    #print(model)
    model = load_checkpoint(model,args.model_path)
    print(model.initial_block)

    prediction(model,args.image_path)