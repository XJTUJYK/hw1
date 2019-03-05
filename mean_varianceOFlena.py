import numpy as np
from PIL import Image
lena = Image.open("C:\\Users\\lenovo\\Desktop\\1\\lena.bmp")
lena_array=np.array(lena)
sum1=lena_array.sum() #sum
array_square=lena_array*lena_array   #square
[X,Y]=np.shape(lena_array)
N=X*Y
mean=sum1/N
lena_array2=(lena_array-mean)*(lena_array-mean)
sum2=lena_array2.sum()
var=sum2/N
print(mean,var)