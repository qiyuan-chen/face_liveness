from torchvision.datasets import ImageFolder
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

# 将图片根据json转移到对应文件夹


def move_image_to_label_folder(agrs=None):
    # 将数据集中的图片转移到对应的文件夹中
    # figs_path = os.path.join(args.dataset_path, args.task_id)
    # json_file = os.path.join(args.dataset_path, str(args.task_id)+'.json')
    dataset_path = './data'
    task_id = 0
    figs_path = os.path.join(dataset_path, str(task_id))
    json_file_path = os.path.join(dataset_path, str(task_id)+'.json')
    json_file = json.load(open(json_file_path))
    for filename in tqdm(os.listdir(figs_path)):
        try:
            label = json_file.get(filename, None)
            if not os.path.exists(os.path.join(figs_path, label)):
                os.makedirs(os.path.join(figs_path, label))
            label_path = os.path.join(figs_path, label)
            os.replace(os.path.join(figs_path, filename),
                       os.path.join(label_path, filename))
        except:
            print("file do not exist")
            continue

move_image_to_label_folder()