import os
import cv2
import glob
import tqdm
import json
import shutil
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', '-e', default=1,type=int)
parser.add_argument('--model', '-m', default=1,type=str)
args = parser.parse_args()

os.makedirs(f'result({args.model})',exist_ok=True)
f = open(f'result({args.model})/result.txt','w')

if args.model == 'segnet':
    output_name = 'SegNet'
elif args.model == 'unet':
    output_name = 'UNet'
elif args.model == 'deeplab_v1':
    output_name = 'DeepLab_v1'
elif args.model == 'deeplab_v3+':
    output_name = 'DeepLab_v3_plus'
else:
    raise Exception('unexpected model name')

def pre_del_dir():
    for d in [f'result({args.model})/figure',f'low_performance_80({args.model})']:
        try:
            shutil.rmtree(d)
        except:
            pass
    

def calDSC(predict,target):
    eps = 0.0001
    predict = predict.astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.dot(predict.reshape(-1),target.reshape(-1))
    union = predict.sum() + target.sum() + eps
    dsc = (2 * intersection + eps) / union
    return dsc

def read_mesure_imgs(model_num):
    
    predict_paths = glob.glob(f'output/{output_name}/{args.model}_{model_num}/*.png')
    dice_coefficient_info = []
    for fpath in tqdm.tqdm(predict_paths):
        name = os.path.split(fpath)[1].replace('.png','')
        target_path = os.path.join('data/yao/test/masks',os.path.split(fpath)[1])
        im_predict = np.array(Image.open(fpath))
        im_target = np.array(Image.open(target_path))
        dsc = calDSC(im_predict,im_target)
        dice_coefficient_info.append([name,dsc])
    dice_coefficient_info.sort(key=lambda x: x[1])
    os.makedirs('json',exist_ok=True)
    json.dump(dice_coefficient_info,open(f'json/{args.model}_DSC_{model_num}.json','w'),indent=2)

def analyze(model_num):
    dice_coefficient_info = json.load(open(f'json/{args.model}_DSC_{model_num}.json','r'))
    dice_coefficients = [x[1] for x in dice_coefficient_info]
    f.write('mean:{:.4f}\t'.format(np.mean(dice_coefficients)))
    f.write('std:{:.4f}\t'.format(np.std(dice_coefficients)))
    # print(np.mean(dice_coefficients))
    # print(np.std(dice_coefficients))

def draw(model_num):
    os.makedirs(f'result({args.model})/figure',exist_ok=True)
    dice_coefficient_info = json.load(open(f'json/{args.model}_DSC_{model_num}.json','r'))
    dice_coefficients = [x[1] for x in dice_coefficient_info]
    less80 = [x for x in dice_coefficients if x<0.80]
    less90 = [x for x in dice_coefficients if x<0.90]
    num_less80 = len(less80)
    num_less90 = len(less90)
    str_less80 = '{:>3} samples less than 0.80 {:>6.2f}%\t'.format(num_less80,num_less80/len(dice_coefficients)*100)
    str_less90 = '{:>3} samples less than 0.90 {:>6.2f}%\t'.format(num_less90,num_less90/len(dice_coefficients)*100)
    f.write(str_less80)
    f.write(str_less90)
    # print(str_less80)
    # print(str_less90)
    plt.clf()
    plt.ylim(0.0,1.0)
    plt.scatter(range(len(dice_coefficients)),dice_coefficients,marker='x')
    plt.title(str_less80+'\n'+str_less90)
    plt.savefig(f'result({args.model})/figure/training_result_{model_num}.png',dpi=300)

def fetch_low_80_performance_data(model_num):
    dice_coefficient_info = json.load(open(f'json/{args.model}_DSC_{model_num}.json','r'))
    names = [x[0] for x in dice_coefficient_info][:]
    os.makedirs(f'low_performance_80({args.model})/{model_num}',exist_ok=True)
    for i,name in tqdm.tqdm(enumerate(names)):
        ori_path = os.path.join('data/yao/test/imgs',name+'.jpg')
        target_path = os.path.join('data/yao/test/masks',name+'.png')
        predict_path = os.path.join(f'output/{output_name}/{args.model}_{model_num}',name+'.png')
        img = np.array(Image.open(ori_path))
        target_mask = np.array(Image.open(target_path))
        predict_mask = np.array(Image.open(predict_path))
        delta_mask = target_mask-predict_mask
        union_mask = target_mask*predict_mask
        
        img[:,:,0][union_mask==1]=0
        img[:,:,1][union_mask==1]=255
        img[:,:,2][union_mask==1]=0

        img[:,:,0][delta_mask==1]=0
        img[:,:,1][delta_mask==1]=0
        img[:,:,2][delta_mask==1]=255
        
        img[:,:,0][delta_mask==255]=255
        img[:,:,1][delta_mask==255]=0
        img[:,:,2][delta_mask==255]=0

        # tmp_img = Image.fromarray(target_mask*255)
        # tmp_img.save('mask_output.png')
        save_path = os.path.join(f'low_performance_80({args.model})/{model_num}','{}_{}.png'.format(i,name))
        pil_img = Image.fromarray(img)
        pil_img.save(save_path)
        

def main():
    pre_del_dir()
    for epoch in range(1,args.epochs+1):
        f.write('{:>2} '.format(epoch))
        read_mesure_imgs(epoch)
        analyze(epoch)
        draw(epoch)
        fetch_low_80_performance_data(epoch)
        f.write('\n')
    f.close()

if __name__ == "__main__":
    main()