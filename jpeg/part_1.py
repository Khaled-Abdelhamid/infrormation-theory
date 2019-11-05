from helpers import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# call the image and convert it into gray scale bitmap
oimage = Image.open("t2.bmp").convert('LA')
oimage.save('t2.png')

gim = plt.imread('t2.png')
gray = rgb2gray(gim)*255
rows,cols=gray.shape

plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)


# quantiation tables

Q8_1=[[1,1,1,1,1,2,2,4],
    [1,1,1,1,1,2,2,4],
    [1,1,1,1,2,2,2,4],
    [1,1,1,1,2,2,4,8],
    [1,1,2,2,2,2,4,8],
    [2,2,2,2,2,4,8,8],
    [2,2,2,4,4,8,8,16],
    [4,4,4,4,8,8,16,16]]

Q8_2=[[1,2,4,8,16,32,64,128],
    [2,4,4,8,16,32,64,128],
    [4,4,8,16,32,64,128,128],
    [8,8,16,32,64,128,128,256],
    [16,16,32,64,128,128,256,256],
    [32,32,64,128,128,256,256,256],
    [64,64,128,128,256,256,256,256],
    [128,128,128,256,256,256,256,256]]

Q8_3= [
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]]

Q4_1=[[1,1,2,4],
     [1,2,2,4],
     [2,2,2,4],
     [4,4,4,8]]


Q4_2=[[16,16,32,64],
     [16,32,32,64],
     [22,32,32,64],
     [64,64,64,128]]

# do the same qunatization level in 8y 8 but repeat every element twice

Q16_1 = np.repeat(np.repeat(Q8_1, 2, axis=0), 2, axis=1)

Q16_2 = np.repeat(np.repeat(Q8_2, 2, axis=0), 2, axis=1)

frows,fcols = 4,4 #choose the frame size
Q=Q1_2     #choose quantization matrix

code,huffman = encode(gray,frows,fcols,Q) # return the encoded image
recovered= decode(code,huffman,rows,cols,frows,fcols,Q)
plt.imshow(recovered, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
# get the least square error and divide it over the sum square of the image
print("the error  the image is : ",100*error(gray,recovered)/np.sum(np.square(gray)),"%")