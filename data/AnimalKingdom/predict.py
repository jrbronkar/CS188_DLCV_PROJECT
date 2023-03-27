""" How to use C3D network. """
import numpy as np

import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D
import cv2
from torchvision import io as visionio
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import torchvision.models as models
import torch.nn as nn 
import os

def main():
    """
    Main function.
    """
    # get C3D pretrained model
    net = C3D()
    net.load_state_dict(torch.load('c3d.pickle'))
    net.cuda()
    net.eval()
    for video_path in os.listdir("./dataset"):
        vid = visionio.read_video("./dataset/"+video_path)[0]
        vid_features = torch.zeros(((len(vid)+7)//8,4096))
        for i in range(8,len(vid)-7,8):
            clip = vid[i-8:i+8]
            clip = torch.permute(clip, (0, 3, 1, 2))  
            # resize clip
            clip = transforms.Resize(224)(clip)
            # centrally crop (outputs 224x224 clip)
            clip = fn.center_crop(clip, output_size=[224])

            clip = torch.permute(clip, (1, 0, 2, 3))  # ch, fr, h, w
            clip = clip[None, :]
            clip = clip.float()
            clip = clip.cuda()
            with torch.no_grad():
                vid_features[i//8] = net(frame)
        torch.save(vid_features,"./C3D-features/"+video_path+".pt")
        torch.cuda.empty_cache()


# entry point
if __name__ == '__main__':
    main()
