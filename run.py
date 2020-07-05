import os
import json
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from model import *
from utils import get_range_limited_float_type,dice_coeff,train,predict
from utils import timewrapper,setup_logger,ToLabel,NLLLOSS2d_logSoftmax
from torchvision.transforms import Compose,Normalize,ToTensor,ToPILImage

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode',type=str,choices=['train','predict'],required=True)
    parser.add_argument('--config-path',type=str,default='config/cfg.json')
    parser.add_argument('--gpu-ids',type=int,nargs='+',default=0,dest='gpu_ids')
    parser.add_argument('--state',type=int,default=1,dest='state')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=2,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-p', '--port', type=int,default=10001,
                        help='Visualization port', dest='port')
    parser.add_argument('-w', '--worker-num', type=int, default=8,
                        help='Dataloader worker number', dest='num_workers')
    parser.add_argument('-c', '--class-num', type=int, default=2,
                        help='class number', dest='class_num')
    parser.add_argument('-v', '--valid-percent', type=get_range_limited_float_type(0,100), default=10.0,
                        help='Percent of the data that is used as validation (0-100)', dest='valid_percent')

    args = parser.parse_args()

    assert os.path.exists(args.config_path),'config json not exists'
    with open(args.config_path,'r') as f:
        config = json.load(f)

    for arg in vars(args):
        config[arg]=getattr(args,arg)
    
    if isinstance(config['gpu_ids'],int):
        config['gpu_ids'] = [config['gpu_ids']]
    config['gpu_ids'] = list(set(config['gpu_ids']))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config['device'] == 'cuda':
        gpu_num = torch.cuda.device_count()
        assert len(config['gpu_ids'])!=0,'unexpected gpu number'
        for gpu_id in config['gpu_ids']:
            assert gpu_id>=0 and gpu_id<gpu_num,'invalid gpu id input'

    config['input_transform'] = Compose([
        # CenterCrop(256),
        ToTensor(),
        # The mean and std result from data statistics of ImageNet dataset, you should fill corresponding mean and std here.
        # Normalize([.485, .456, .406], [.229, .224, .225]), 
    ])

    config['target_transform'] = Compose([
        # CenterCrop(256),
        ToLabel(),
        # Relabel(255, 21),
    ])
            
    return config


def main(cfg):
    if cfg['model'] == 'segnet':
        net = SegNet(classes=cfg['class_num'])
    elif cfg['model'] == 'unet':
        net = UNet(n_channels=3, n_classes=cfg['class_num'], bilinear=True)
    elif cfg['model'] == 'deeplab_v1':
        net = DeepLab_v1(in_dim=3,out_dim=2)
    elif cfg['model'] == 'deeplab_v2':
        net = DeepLab_v2(
            n_classes=cfg['class_num'], 
            n_blocks=[3, 4, 23, 3], 
            atrous_rates=[6, 12, 18, 24]
        )
    elif cfg['model'] == 'deeplab_v3':
        net = DeepLab_v3(
            n_classes=2,
            n_blocks=[3, 4, 23, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,
        )
    elif cfg['model'] == 'deeplab_v3+':
        net = DeepLab_v3_plus(
            n_classes=2,
            n_blocks=[3, 4, 23, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,
        )
    else:
        raise Exception('model {} not available'.format(cfg['model']))
    
    if cfg['device']=='cuda':
        if len(cfg['gpu_ids'])==1:
            torch.cuda.set_device(cfg['gpu_ids'][0])
            net = net.cuda()
        else:
            net = net.cuda()
            net = nn.DataParallel(net,device_ids=cfg['gpu_ids'])

    if cfg['mode'] == 'train':
        train(cfg,net)
    elif cfg['mode'] == 'predict':
        predict(cfg,net,'checkpoints/{}_{}.pth'.format(cfg['model'],cfg['state']))
        

if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
    