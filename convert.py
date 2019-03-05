from PIL import Image
import cv2
import numpy as np
import math
def nearest(image,x,y):
    image=Image.open(image)
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
    image=Image.open(image)
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
#构造bicubic函数从而进行双三次线性插值
def bicubic_prefunction(x):
    x = np.abs(x)
    if 0 <= x < 1:
        return 1 - 2 * x * x + x * x * x
    if 1 <= x < 2:
        return 4 - 8 * x + 5 * x * x - x * x * x
    else:
        return 0
def bicubic(t, m, n):
    img = cv2.imread(t)
    height, width, channels = img.shape
    emptyImage = np.zeros((m, n, channels), np.uint8)
    sh = m / height
    sw = n / width
    for i in range(m):
        for j in range(n):
            x = i / sh
            y = j / sw
            p = (i + 0.0) / sh - x
            q = (j + 0.0) / sw - y
            x = int(x) - 2
            y = int(y) - 2
            A = np.array([
                [bicubic_prefunction(1 + p), bicubic_prefunction(p), bicubic_prefunction(1 - p), bicubic_prefunction(2 - p)]
            ])
            if x >= m - 3:
                m - 1
            if y >= n - 3:
                n - 1
            if x >= 1 and x <= (m - 3) and y >= 1 and y <= (n - 3):
                B = np.array([
                    [img[x - 1, y - 1], img[x - 1, y],
                     img[x - 1, y + 1],
                     img[x - 1, y + 1]],
                    [img[x, y - 1], img[x, y],
                     img[x, y + 1], img[x, y + 2]],
                    [img[x + 1, y - 1], img[x + 1, y],
                     img[x + 1, y + 1], img[x + 1, y + 2]],
                    [img[x + 2, y - 1], img[x + 2, y],
                     img[x + 2, y + 1], img[x + 2, y + 1]],

                ])
                C = np.array([
                    [bicubic_prefunction(1 + q)],
                    [bicubic_prefunction(q)],
                    [bicubic_prefunction(1 - q)],
                    [bicubic_prefunction(2 - q)]
                ])
                blue = np.dot(np.dot(A, B[:, :, 0]), C)[0, 0]
                green = np.dot(np.dot(A, B[:, :, 1]), C)[0, 0]
                red = np.dot(np.dot(A, B[:, :, 2]), C)[0, 0]

                # ajust the value to be in [0,255]
                def adjust(value):
                    if value > 255:
                        value = 255
                    elif value < 0:
                        value = 0
                    return value

                blue = adjust(blue)
                green = adjust(green)
                red = adjust(red)
                emptyImage[i, j] = np.array([blue, green, red], dtype=np.uint8)
    data = Image.fromarray(emptyImage)
    data.show()

"""
def bicubic(image,x,y):
    image=Image.open(image)
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
"""
nearest("C:\\Users\\lenovo\\Desktop\\1\\lena.bmp",2048,2048)
bilinear("C:\\Users\\lenovo\\Desktop\\1\\lena.bmp",2048,2048)
bicubic("C:\\Users\\lenovo\\Desktop\\1\\lena.bmp",2048,2048)