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
    # get pretrained model
    net = models.vgg16(pretrained=True)
    net.classifier = nn.Sequential(
        *list(net.classifier.children())[:-1],
    )

    net.cuda()
    net.eval()
    for video_path in os.listdir("./dataset"):
        vid = visionio.read_video("./dataset/"+video_path)[0]
        vid_features = torch.zeros(((len(vid)+7)//8,4096))
        for i in range(0,len(vid),8):
            frame = vid[i]
            # resize frame to 224x224
            frame = torch.permute(frame, (2, 0, 1))
            # perform resize
            frame = transforms.Resize(224)(frame)
            # get 224x224 center region. This method prevents distortion.
            frame = fn.center_crop(frame, output_size=[224])
            frame = frame[None, :]
            frame = frame.float()
            frame = frame.cuda()
            with torch.no_grad():
                vid_features[i//8] = net(frame)
        torch.save(vid_features,"./vgg-features/"+video_path+".pt")
        torch.cuda.empty_cache()


# entry point
if __name__ == '__main__':
    main()
