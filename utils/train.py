import os
import json
import torch
import datetime
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from model import SegNet
from utils import get_range_limited_float_type,dice_coeff
from utils import timewrapper,setup_logger,TrainingDataset,ToLabel,NLLLOSS2d_logSoftmax,CrossEntropyLoss2d
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader,random_split

def eval(cfg,net,loader,device):
    target_type = torch.float32 if cfg['class_num'] == 1 else torch.long
    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val,desc='Validation ',unit='batch',leave=False) as pbar:
        for iter,(imgs,targets) in enumerate(loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                predict = net(imgs)

            if cfg['class_num'] > 1:
                weight = torch.tensor(cfg['weight']).to(device)
                tot += F.cross_entropy(predict,targets,weight).item()
            else:
                mask = torch.sigmoid(predict)
                mask = (mask>0.5).float()
                tot += dice_coeff(mask,targets).item()
            pbar.update()
    print('') # for better display
    
    return tot/n_val

def train(cfg,net):
    lr = cfg['lr']
    mode = cfg['mode']
    device = cfg['device']
    weight = cfg['weight']
    img_dir = cfg['train_img_dir']
    mask_dir = cfg['train_mask_dir']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    input_transform = cfg['input_transform']
    target_transform = cfg['target_transform']
    model_name = net.__class__.__name__

    current_time = datetime.datetime.now()
    logger_file = os.path.join('log',mode,'{} {} lr {} bs {} ep {}.log'.
                    format(model_name,current_time.strftime('%Y%m%d %H:%M:%S'),
                            cfg['lr'],cfg['batch_size'],cfg['epochs']))
    logger = setup_logger(f'{model_name} {mode}',logger_file)

    dataset = TrainingDataset(img_dir,mask_dir,input_transform,target_transform,logger)
    loader = DataLoader(dataset)
    n_val = int(len(dataset)*cfg['valid_percent']/100)
    n_train = len(dataset)-n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    display_weight = weight
    weight = torch.tensor(weight)
    if device == 'cuda':
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight=weight)
    

    # if cfg['model'].startswith('Seg'):
    #     optimizer = SGD(net.parameters(),1e-3,.9)
    # else:
    #     optimizer = Adam(net.parameters())

    # default lr = 0.001  betas = (0.9,0.999)  eps=1e-08  weight_decay = 0
    optimizer = Adam(net.parameters(),lr=cfg['lr'],betas=(0.9,0.999))

    # logging.info(f'''Starting training:
    #     Epochs:          {cfg['epochs']}
    #     Batch size:      {cfg['batchsize']}
    #     Learning rate:   {cfg['lr']}
    #     Training size:   {cfg['n_train']}
    #     Validation size: {cfg['n_val']}
    #     # Checkpoints:     {cfg['save_cp']}
    #     Device:          {cfg['device']}
    #     # Images scaling:  {cfg['img_scale']}
    # ''')
    logger.info(f'''Starting training:
        Model:           {net.__class__.__name__}
        Epochs:          {cfg['epochs']}
        Batch size:      {cfg['batch_size']}
        Learning rate:   {cfg['lr']}
        Training size:   {n_train}
        Weight:          {display_weight}
        Validation size: {n_val}
        Device:          {device}
    ''')

    iter_num = 0
    train_batch_num = len(train_dataset)//batch_size
    for epoch in range(1,cfg['epochs']+1):
        net.train()
        epoch_loss = []
        logger.info('epoch[{}/{}]'.format(epoch,cfg['epochs']))
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch,cfg['epochs']),unit='imgs') as pbar:
            for iter, (imgs,targets) in enumerate(train_loader):

                imgs = imgs.to(device)
                targets = targets.to(device)

                predict = net(imgs)
                loss = criterion(predict,targets)

                loss_item = loss.item()
                epoch_loss.append(loss_item)
                pbar.set_postfix(**{'loss (batch)':loss_item})
                
                optimizer.zero_grad()
                loss.backward()
                # make sure that gradient value falls in the range(-clip_value,clip_value)
                # nn.utils.clip_grad_value_(net.parameters(), 0.1) 
                optimizer.step()

                pbar.update(imgs.shape[0])
                iter_num += 1
                if iter_num % (train_batch_num//10) == 0:
                    val_score = eval(cfg,net,val_loader,device)

                    if cfg['class_num']>1:
                        logger.info('Validation cross entropy: {:.6f}'.format(val_score))
                    else:
                        logger.info('Validation Dice Coeff: {:.6f}'.format(val_score))
        if True:
            if not os.path.exists(cfg['checkpoint_dir']):
                os.makedirs(cfg['checkpoint_dir'])
                logger.info('Created checkpoint directory:{}'.format(cfg['checkpoint_dir']))
            torch.save(net.state_dict(),os.path.join(cfg['checkpoint_dir'],"{}_{}.pth".format(cfg['model'],epoch)))
            logger.info(f'Checkpoint {epoch} saved!')