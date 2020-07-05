import os
import glob
import torch
import shutil
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from model import SegNet
from os.path import split,splitext
from utils import setup_logger,is_image_file,TestingDataset
from torch.utils.data import DataLoader

def predict(cfg,net,model_path):
    mode = cfg['mode']
    device = cfg['device']
    class_num = cfg['class_num']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    test_img_dir = cfg['test_img_dir']

    model_name = net.module.__class__.__name__  \
                if isinstance(net,nn.DataParallel) \
                else net.__class__.__name__

    output_dir = os.path.join(cfg['output_dir'],model_name,splitext(split(model_path)[1])[0])
    #WARNING: this will remove the orignal directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir,exist_ok=True)

    current_time = datetime.datetime.now()
    logger_file = os.path.join('log',mode,'{} {}.log'.
                    format(model_name,current_time.strftime('%Y%m%d %H:%M:%S')))
    logger = setup_logger(f'{model_name} {mode}',logger_file)

    # in_files = glob.glob(test_img_dir+'/*')
    # in_files = [x for x in in_files if is_image_file(x)]

    dataset = TestingDataset(test_img_dir,cfg['input_transform'],logger)
    loader = DataLoader(dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    net.load_state_dict(torch.load(model_path,map_location=device))

    net.eval()
    with torch.no_grad():
        for iter,(imgs,filenames) in tqdm(enumerate(loader)):
            imgs = imgs.to(device)
            predict = net(imgs)
            if class_num > 1:
                probs = F.softmax(predict,dim=1)
            else:
                probs = torch.sigmoid(predict)
            # print(probs.shape)
            
            masks = torch.argmax(probs,dim=1).cpu().numpy()
            # print(masks.shape)
            # exit(0)
            for i,file_name in enumerate(filenames):
                mask = masks[i]
                fpath = os.path.join(output_dir,splitext(file_name)[0]+'.png')
                mask = Image.fromarray(mask.astype(np.uint8))
                mask.save(fpath)
            
        

