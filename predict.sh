#!/bin/bash  
model=deeplab_v1
e=30

for i in `seq $e`;  
do   
python run.py --model $model --mode predict --state $i
done

python measure.py --model $model --epochs $e
# python measure.py --model unet --epochs 30
# python measure.py --model deeplab_v3+ --epochs 5
