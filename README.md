# LOSS
## Dice Loss
**Dice Coefficent**

<img src="https://latex.codecogs.com/gif.latex?DSC(A,B)=\frac{2|A\cap&space;B|}{|A|&plus;|B|}" title="DSC(A,B)=\frac{2|A\cap B|}{|A|+|B|}" align=right />


**Dice Loss**
$$
d = 1-DSC = 1 - \frac{2|A\cap B|}{|A|+|B|}
$$

**Laplace Smoothing**
$$
L_s = 1-\frac{2|A\cap B|+1}{|A|+|B|+1}
$$

Benefits:
1. In case both $|A|$ and $|B|$ equal to 0, divided by zero,
2. Alleviate overfitting


## IOU LOSS
$$
IOU(A,B)=\frac{|A\cap B|}{|A\cup B|}\\
$$

## Focal Loss
$$
\alpha y\left(1-p\right)^{\gamma} \log p+(1-\alpha)\left(1-y\right) p^{\gamma} \log \left(1-p\right)
$$
