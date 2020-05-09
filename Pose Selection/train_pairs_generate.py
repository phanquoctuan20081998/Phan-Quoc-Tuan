
import os
import os.path as osp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

criterionL1 = nn.L1Loss()
transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def last_5chars(x):
    return(x[-8:-3])

def generate_test_pair(source_dir, out_dir):
    file = open(out_dir, 'w')
    video_input_files = sorted(os.listdir(osp.join(source_dir, 'image')), key=last_5chars)
    video_gt_files = sorted(os.listdir(osp.join(source_dir, 'wraped')), key=last_5chars)

    for video_file in video_input_files:
        if video_file.startswith('.'):
            continue

        frames = sorted(os.listdir(os.path.join(source_dir, 'image', video_file)), key=last_5chars)
        
        losses = []
        for frame in frames:
            if frame.startswith('.'):
                continue

            in_frame = Image.open(osp.join(source_dir, 'image', video_file, frame))
            in_frame = transform(in_frame)

            gt_frame = Image.open(osp.join(source_dir, 'wraped', video_file, frame))
            gt_frame = transform(gt_frame)

            losses.append(criterionL1(gt_frame, in_frame).numpy())
            print(frame, ' = ', criterionL1(gt_frame, in_frame).numpy())
        
        index = losses.index(min(losses))

        file.write('%s %s\n' % (video_file, str(index).zfill(5)))

    file.close()

source_dir = 'data'
out_dir = 'data/test_pair.txt'
generate_test_pair(source_dir, out_dir)