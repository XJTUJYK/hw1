from PIL import Image
import numpy as np
from math import floor


def change_gray_level(str,reduce_level):
    lena = Image.open(str)
    lena_array=np.array(lena)
    [x,y]=np.shape(lena_array)
    if reduce_level<0:
        reduce_level=0
    if reduce_level>8:
        reduce_level=8
    else:pass
    real_reduce=pow(2,reduce_level) #实际降低灰度的值
    tran=np.zeros([x,y])
    result=np.zeros([x,y])
    for i in range(0,x-1):
        for j in range(0,y-1):
            tran[i,j]= floor(lena_array[i,j] / real_reduce)#除以实际降低灰度值向下取整
            result[i,j]=tran[i,j]*real_reduce#再乘实际灰度值
    print(result)
    data = Image.fromarray(result)
    data.show()

for i in range(0,8):
    change_gray_level("C:\\Users\\lenovo\\Desktop\\1\\lena.bmp",i)



