import torch.utils.data as data
import os
import os.path
import glob
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_shadow_pair_dataset(dir):
    images = []
    srimages = []
    print(dir)
    print(os.walk(dir))
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                srimpath = path.replace("original_image","sr_image")
                images.append(path)
                srimages.append(srimpath)

    return images,srimages


def rgb_loader(path):
    return Image.open(path).convert('RGB')

def gray_loader(path):
    return Image.open(path).convert('L')

class CustomShadowPairDataset(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 rgb_loader=rgb_loader, gray_loader=gray_loader):
        imgs,srimages = make_shadow_pair_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.srimgs = srimages
        self.transform = transform
        self.target_transform = target_transform
        self.rgb_loader = rgb_loader
        self.gray_loader = gray_loader

    def __getitem__(self, index):
        imgpath = self.imgs[index]
        srimgpath = self.srimgs[index]

        img = self.rgb_loader(imgpath)
        srimg = self.gray_loader(srimgpath)

        if self.transform is not None:
            img = self.transform(img)
            srimg = self.transform(srimg)

        return img, srimg

    def __len__(self):
        return len(self.imgs)