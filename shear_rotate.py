import numpy as np
from PIL import Image
import math


def shear(image, k):
    image = Image.open(image)
    image_array = np.array(image)
    [m,n] = np.shape(image_array)
    t=int(m*k+n)
    new_array = np.zeros([t, n])
    for i in range(0,m-1):
        for j in range(0,n-1):
            x=int(i+j*k)
            new_array[x,j]=image_array[i,j]
    data = Image.fromarray(new_array)
    data.show()
    return data
def rotation(image,angle):
    image = Image.open(image)
    image_array = np.array(image)
    h = image.height
    w = image.width
    anglePi = angle * math.pi / 180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)
    X1 = math.ceil(abs(0.5 * h * cosA + 0.5 * w * sinA))
    X2 = math.ceil(abs(0.5 * h * cosA - 0.5 * w * sinA))
    Y1 = math.ceil(abs(-0.5 * h * sinA + 0.5 * w * cosA))
    Y2 = math.ceil(abs(-0.5 * h * sinA - 0.5 * w * cosA))
    H = int(2 * max(Y1, Y2))
    W = int(2 * max(X1, X2))
    new_array = np.zeros([W+1,H+1])
    for i in range(h):
        for j in range(w):
            x = int(cosA * i - sinA * j - 0.5 * w * cosA + 0.5 * h * sinA + 0.5 * W)
            y = int(sinA * i + cosA * j - 0.5 * w * sinA - 0.5 * h * cosA + 0.5 * H)
            new_array[x, y] = image_array[i, j]
    data = Image.fromarray(new_array)
    data.show()
    return data
def nearest(image,x,y):
    image_array=np.array(image)
    [m,n]=np.shape(image_array)
    new_array=np.zeros([x,y])
    deth1=m/x
    deth2=n/y
    for i in range(0,x-1):
        for j in range(0,y-1):
            a=round((i+1)*deth1)
            b=round((j+1)*deth2)
            new_array[i,j]=image_array[a-1,b-1]
    data = Image.fromarray(new_array)
    data.show()
def bilinear(image,x,y):
    image_array=np.array(image)
    [m,n]=np.shape(image_array)
    new_array=np.zeros([x,y])
    deth1=m/x
    deth2=n/y

    # f(x , y) = f(X + u, Y + v) =f (X, Y)  * (1 - u) * (1 - v) + f(X, Y + 1) * (1 - u) * v + f(X + 1, Y) * u * (1 - v) + f (X + 1, Y + 1) * u * v;
    for i in range(0,x-1):
        for j in range(0,y-1):
            X=(i+1)*deth1
            Y=(j+1)*deth2
            intX=int(X)
            intY=int(Y)
            u=X-intX
            v=Y-intY
            new_array[i,j] =image_array[intX-1, intY-1] * (1 - u) * (1 - v) + image_array[intX-1, intY ] * (1 - u) * v + image_array[intX, intY-1] * u * (1 - v) + image_array [intX , intY ] * u * v
    data = Image.fromarray(new_array)
    data.show()
def bicubic_prefunction(x):
    abs_x=abs(x)
    if abs_x<1 or abs==1:
        W=1.5*pow(abs_x,3)-2.5*pow(abs_x,2)+1
        return W
    elif abs_x<2 or abs_x>1:
        W=-0.5*pow(abs_x,3)+2.5*pow(abs_x,2)-4*abs_x+2
        return W
    else:
        W=0
        return 0
def bicubic(image,x,y):
    image_array=np.array(image)
    [m,n]=np.shape(image_array)
    F=np.zeros([x,y])
    deths1=m/x
    deths2=n/y
    for i in range(0,x-1):
        for j in range(0,y-1):
            X=deths1*(i+1)
            Y=deths2*(j+1)
            intX=int(X)
            intY=int(Y)
            u=X-intX
            v=Y-intY
            A=np.matrix([[bicubic_prefunction(1+v),bicubic_prefunction(v),bicubic_prefunction(1-v),bicubic_prefunction(2-v)]])
            B=np.matrix([[image_array[intX-3,intY-3],image_array[intX-3,intY-2],image_array[intX-3,intY-1],image_array[intX-3,intY]],
                    [image_array[intX-2,intY-3],image_array[intX-2,intY-2],image_array[intX-2,intY-1],image_array[intX-2,intY]],
                    [image_array[intX-1,intY-3],image_array[intX-1,intY-2],image_array[intX-1,intY-1],image_array[intX-1,intY]],
                    [image_array[intX,intY-3],image_array[intX,intY-2],image_array[intX,intY-1],image_array[intX,intY]]])
            test=np.matrix([bicubic_prefunction(1+u),bicubic_prefunction(u),bicubic_prefunction(1-u),bicubic_prefunction(2-u)])
            C=test.transpose()
            F[i,j]=float(A*B*C)
    data=Image.fromarray(F)
    data.show()
def rotate_usepack(image, angle):
    img2=Image.open(image)
    img1 = img2.resize((2048, 2048))
    img3 = img1.rotate(angle)

    img3.show()

#rotate_usepack("C:\\Users\\lenovo\\Desktop\\1\\lena.bmp",30)
data2 = shear("C:\\Users\\lenovo\\Desktop\\1\\elain1.bmp",1.5)
nearest(data2,2048,2048)
bilinear(data2,2048,2048)
bicubic(data2,2048,2048)
data4 = rotation("C:\\Users\\lenovo\\Desktop\\1\\elain1.bmp",30)
nearest(data4,2048,2048)
bilinear(data4,2048,2048)
bicubic(data4,2048,2048)
data1 = shear("C:\\Users\\lenovo\\Desktop\\1\\lena.bmp",1.5)
nearest(data1,2048,2048)
bilinear(data1,2048,2048)
bicubic(data1,2048,2048)
data3 = rotation("C:\\Users\\lenovo\\Desktop\\1\\lena.bmp",30)
nearest(data3,2048,2048)
bilinear(data3,2048,2048)
bicubic(data3,2048,2048)



