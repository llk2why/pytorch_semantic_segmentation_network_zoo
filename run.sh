# python run.py --model segnet --mode train -l 0.001 -b 5 -e 30 --gpu-id 0
# python run.py --model unet --mode train -l 0.001 -b 5 -e 30 --gpu-id 0
python run.py --model deeplab_v1 --mode train -l 0.001 -b 5 -e 30 --gpu-id 0
