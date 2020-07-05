

# Pytorch Semantic Segment Model Zoo

## Brief Introduction
This repository is organized for the purpose to gather classic  sematic segmentation models and provides a unified framework to train&test models, and each model is pluggable so that it's easy to "plug" in or out a specific model .  Many models are borrowed from other repositories, and there are some modifications for better readability. It is assumed that each image has three channels.


## LOSS
The following losses may not be implemented in the codes, and they are listed to give a better 

### Dice Loss
#### **Dice Coefficent**

<div align=center>
<img src="https://latex.codecogs.com/gif.latex?DSC(A,B)=\frac{2|A\cap&space;B|}{|A|&plus;|B|}" 
title="DSC(A,B)=\frac{2|A\cap B|}{|A|+|B|}" /> 
</div>
#### **Dice Loss**

<div align=center>
<img src="https://latex.codecogs.com/gif.latex?d&space;=&space;1-DSC&space;=&space;1&space;-&space;\frac{2|A\cap&space;B|}{|A|&plus;|B|}" title="d = 1-DSC = 1 - \frac{2|A\cap B|}{|A|+|B|}" />
</div>
#### **Laplace Smoothing**

<div align=center>
<img src="https://latex.codecogs.com/gif.latex?L_s&space;=&space;1-\frac{2|A\cap&space;B|&plus;1}{|A|&plus;|B|&plus;1}" title="L_s = 1-\frac{2|A\cap B|+1}{|A|+|B|+1}" />
</div>


Benefits:
1. In case both $|A|$ and $|B|$ equal to 0, divided by zero,
2. Alleviate overfitting


### IOU LOSS
<div align=center>
<img src="https://latex.codecogs.com/gif.latex?IOU(A,B)=\frac{|A\cap&space;B|}{|A\cup&space;B|}" title="IOU(A,B)=\frac{|A\cap B|}{|A\cup B|}" />
</div>

### Focal Loss
<div align=center>
<img src="https://latex.codecogs.com/gif.latex?\alpha&space;y\left(1-p\right)^{\gamma}&space;\log&space;p&plus;(1-\alpha)\left(1-y\right)&space;p^{\gamma}&space;\log&space;\left(1-p\right)" title="\alpha y\left(1-p\right)^{\gamma} \log p+(1-\alpha)\left(1-y\right) p^{\gamma} \log \left(1-p\right)" />
</div>

