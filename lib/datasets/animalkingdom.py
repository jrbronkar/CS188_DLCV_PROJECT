""" Dataset loader for the AnimalKingdom dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from sklearn.decomposition import PCA

from . import average_to_fixed_length
from core.eval import iou
from core.config import config
import numpy as np

class AnimalKingdom(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(AnimalKingdom, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        self.vid_to_idx = {}
        self.features = None
        
        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        annotations = []
        with open(os.path.join(self.data_dir, '{}.txt'.format(split)),'r') as f:
            for l in f:
                hashtag = l.split("##")
                sentence = hashtag[1].rstrip('\n')
                vid, start,end = hashtag[0].split(" ")
                annotations.append((vid,float(start),float(end),sentence))

        vid_to_durations = {}
        with open(os.path.join(self.data_dir,'ak_vg_duration.json'),'r') as f:
            video_durations = json.load(f)
            for pair in video_durations:
                vid_to_durations[pair["vid"]] = pair["duration"]
        
        anno_pairs = []
        for vid,start,end,sentence in annotations:
            duration = vid_to_durations[vid]
            if start < end:
                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times':[max(start,0),min(end,duration)],
                            'description':sentence,
                        }
                    )
        self.annotations = anno_pairs
    
    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        sentence = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        visual_input, visual_mask = self.get_video_features(video_id)

        # Time scaled to same size
        if config.DATASET.NUM_SAMPLE_CLIPS > 0:
            # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
            visual_input = average_to_fixed_length(visual_input)
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
            s_times = torch.arange(0,num_clips).float()*duration/num_clips
            e_times = torch.arange(1,num_clips+1).float()*duration/num_clips
            overlaps = iou(torch.stack([s_times[:,None].expand(-1,num_clips),
                                        e_times[None,:].expand(num_clips,-1)],dim=2).view(-1,2).tolist(),
                           torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips,num_clips)

        # Time unscaled NEED FIXED WINDOW SIZE
        else:
            num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
            raise NotImplementedError
            # torch.arange(0,)

        item = {
            'visual_input': visual_input,
            'vis_mask': visual_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'map_gt': torch.from_numpy(overlaps),
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        features = torch.load(os.path.join(self.data_dir, "c3d-pytorch/vgg-features/"+vid+".mp4.pt"))
        #with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
        #    features = torch.from_numpy(f[vid]['c3d_features'][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask
