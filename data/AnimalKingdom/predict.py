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

def get_sport_clip(clip_name, verbose=False):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?
    
    Parameters
    ----------
    clip_name: str
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    clip = sorted(glob(join('data', clip_name, '*.png')))
    clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)


def read_labels_from_file(filepath):
    """
    Reads Sport1M labels from file
    
    Parameters
    ----------
    filepath: str
        the file.
        
    Returns
    -------
    list
        list of sport names.
    """
    with open(filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def main():
    """
    Main function.
    """

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
