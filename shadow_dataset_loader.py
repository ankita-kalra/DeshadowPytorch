import torch.utils.data as data
import os
import os.path
import glob
import torchvision.transforms as transforms
from PIL import Image


def make_dataset(img_path, label_path):
    dataset = []
    for img in glob.glob(os.path.join(img_path, '*.jpg')):
        basename = os.path.basename(img)
        image = os.path.join(img_path, basename)
        label = os.path.join(label_path, basename)
        dataset.append([image, label])
    return dataset

dataset=make_dataset()