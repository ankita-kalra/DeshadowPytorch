import torch.nn as nn
import torch

def concatenate_shadow_matte(outS,outA):
	conv_8=nn.Conv2d(6,1, kernel_size=(1, 1))
	input=torch.cat((outS, outA), 1)
	print(input.shape)
	outFinalMatte = conv_8(input)
	return outFinalMatte