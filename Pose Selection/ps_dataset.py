#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

import os

class PSDataset(data.Dataset):
    def __init__(self, opt):
        super(PSDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # load data list
        vid_names = []
        gts_id = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                vid_name, gt_id = line.strip().split()
                vid_names.append(vid_name)
                gts_id.append(gt_id)

        self.vid_names = vid_names
        self.gts_id = gts_id

    def name(self):
        return "PSDataset"

    def __getitem__(self, index):
        vid_name = self.vid_names[index]
        gt_id = self.gts_id[index]

        vid_gt_path = osp.join(self.root, 'image', vid_name) 
        vid_ip_path = osp.join(self.root, 'wraped', vid_name)       
        
        vid_ip = os.listdir(vid_ip_path)

        ip = []
        gt = []
        for frame_name in vid_ip:
            if frame_name.startswith('.'):
                continue
            frame_ip = Image.open(osp.join(vid_ip_path, frame_name))
            frame_gt = Image.open(osp.join(vid_gt_path, frame_name))

            frame_ip = self.transform(frame_ip) # [-1,1]
            frame_gt = self.transform(frame_gt) # [-1,1]

            ip.append(frame_ip)
            gt.append(frame_gt)

        # input list
        ip_list = ip

        # ground truth list
        gt_list = gt

        # input
        ip = torch.cat(ip, 0)

        # input ground truth
        ip_gt = Image.open(osp.join(vid_gt_path, vid_name + '_' + gt_id + '.jpg'))
        ip_gt = self.transform(ip_gt)

        ip_ip = Image.open(osp.join(vid_ip_path, vid_name + '_' + gt_id + '.jpg'))
        ip_ip = self.transform(ip_ip)
        
        # gt = torch.zeros([20], dtype=torch.long)
        # gt[int(gt_id)] = 1
        gt = torch.tensor(int(gt_id), dtype=torch.long)

        result = {
            'ip':   ip,              # for input
            'ip_list':  ip_list,     # for cal loss
            'gt_list':  gt_list,     # for cal loss
            'ip_gt':    ip_gt,       # for ground truth
            'ip_ip':    ip_ip,       # for ground truth
            'gt':       gt,          # for ground truth
            }

        return result

    def __len__(self):
        return len(self.vid_names)

class PSDataLoader(object):
    def __init__(self, opt, dataset):
        super(PSDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch



