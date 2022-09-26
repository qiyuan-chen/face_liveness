from torchvision.datasets import ImageFolder
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json


def get_label_and_image(agrs=None):
    # figs_path = os.path.join(args.dataset_path, args.task_id)
    # json_file = os.path.join(args.dataset_path, str(args.task_id)+'.json')
    dataset_path = './data'
    task_id = 0
    figs_path = os.path.join(dataset_path, str(task_id))
    json_file_path = os.path.join(dataset_path, str(task_id)+'.json')
    json_file = json.load(open(json_file_path))
    for filename in os.listdir(figs_path):
        image = cv2.imread(os.path.join(figs_path, filename))
        print(image.shape)


data_path = "./data/0/"
transform = transforms.Compose(
    [transforms.RandomResizedCrop([224,224]),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

data_all = ImageFolder(root=data_path, transform=transform)
