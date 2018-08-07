import torch.nn as nn
import torch
from torch.backends import cudnn


def concatenate_shadow_matte(outS, outA):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark=True
    conv_8 = nn.Conv2d(6, 3, kernel_size=(1, 1))
    input = torch.cat((outS, outA), 1).to(device)
    print(input.shape)
    outFinalMatte = conv_8(input)
    return outFinalMatte