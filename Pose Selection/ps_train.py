import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time

from ps_dataset import PSDataLoader, PSDataset
from network import ResNet50

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "PS")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--data_list", default = "train_pair.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 100)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 10000)
    parser.add_argument("--decay_step", type=int, default = 10000)

    opt = parser.parse_args()
    return opt


def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()

def train(opt, train_loader, model):
    model.cuda()
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionCEL = nn.CrossEntropyLoss().cuda()
    
    for param in model.parameters():
        param.requires_grad = True

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        ip = inputs['ip'].cuda()
        ip_list = inputs['ip_list']
        gt_list = inputs['gt_list']
        ip_gt = inputs['ip_gt'].cuda()
        ip_ip = inputs['ip_ip'].cuda()
        gt = inputs['gt'].cuda()

        out = model(ip)
        loss = criterionCEL(out, gt)
        

        # Backprop and perform Adam
        with torch.no_grad():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f' 
                    % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

def main():
    opt = get_opt()
    print(opt)
   
    # create dataset 
    train_dataset = PSDataset(opt)

    # create dataloader
    train_loader = PSDataLoader(opt, train_dataset)

    # Init model
    model = ResNet50()
    if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    train(opt, train_loader, model)
    save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'ps_final.pth'))

if __name__ == "__main__":
    main()