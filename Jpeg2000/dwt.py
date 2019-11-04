import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from helpers import *


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


img = plt.imread('img_3.jpg')
img = rgb2gray(img)

# plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)


def filter(vec,f):
	newvec = np.convolve(vec, f)
	return newvec 

def downsample(vec):
	vec=vec[::2]
	return vec


def upsample(vec):
    length = len(vec)
    newvec = np.zeros((2*length))
    for i in range(length):
        newvec[2*i]=vec[i]  
        
    for i in range(1,len(newvec)-1,2):
        newvec[i]=(newvec[i-1]+newvec[i+1])/2 
    return newvec



	
x = [1,1,1,1,1]
y = [1,1,1,1,1]

ll = []



############################################
#discrete wavelet transform


def dwt(img,n):

	h0=np.array([-1/8 , 1/4 , 3/4 , 1/4 , -1/8])

	h1=np.array([-1/2 , 1 , -1/2])

	rows, columns = img.shape


	

	######################

	ll_1 = np.zeros((img.shape[0], img.shape[1]//2))
	ll_2 = np.zeros((ll_1.shape[0]//2, ll_1.shape[1]))
	


	for row in range(rows):
		ll_1[row,:] = downsample(filter(img[row,:], h0)[:-len(h0)+1])


	
	for column in range(ll_1.shape[1]):
		ll_2 [:,column] = downsample(filter(ll_1[:,column], h0)[:-len(h0)+1])


	img[0:img.shape[0]//2,0:img.shape[1]//2] = ll_2//n[0]
	



	#######################

	lh_1 = np.zeros((img.shape[0], img.shape[1]//2))
	lh_2 = np.zeros((lh_1.shape[0]//2, lh_1.shape[1]))
	
	for row in range(rows):
		lh_1[row,:] = downsample(filter(img[row,:], h0)[:-len(h0)+1])
	
	for column in range(lh_1.shape[1]):
		lh_2 [:,column] = downsample(filter(lh_1[:,column], h1)[:-len(h1)+1])


	img[img.shape[0]//2:,0:img.shape[1]//2] = lh_2//n[1]

	#######################

	hl_1 = np.zeros((img.shape[0], img.shape[1]//2))
	hl_2 = np.zeros((hl_1.shape[0]//2, hl_1.shape[1]))
	


	for row in range(rows):
		hl_1[row,:] = downsample(filter(img[row,:], h1)[:-len(h1)+1])
	
	for column in range(hl_1.shape[1]):
		hl_2 [:,column] = downsample(filter(hl_1[:,column], h0)[:-len(h0)+1])


	img[img.shape[0]//2:,0:img.shape[1]//2] = hl_2//n[2]


	#######################
	hh_1 = np.zeros((img.shape[0], img.shape[1]//2))
	hh_2 = np.zeros((hh_1.shape[0]//2, hh_1.shape[1]))
	
	for row in range(rows):
		hh_1[row,:] = downsample(filter(img[row,:], h0)[:-len(h0)+1])
	
	for column in range(hh_1.shape[1]):
		hh_2 [:,column] = downsample(filter(hh_1[:,column], h0)[:-len(h0)+1])


	img[img.shape[0]//2:,img.shape[1]//2:] = hh_2//n[3]

	#######################


	return img


img = dwt(img,[1,2,3,4])
img[0:img.shape[0]//2,0:img.shape[1]//2] = dwt(img[0:img.shape[0]//2,0:img.shape[1]//2],[1,1,1,1])
img[0:img.shape[0]//4,0:img.shape[1]//4] = dwt(img[0:img.shape[0]//4,0:img.shape[1]//4],[1,1,1,1])

print(img)

one_d = TwoDoneD(img)


###########################################################
run_length_encoded = run_length(one_d)
huffman  = Huffman_encoding()
huffman_encoded = huffman.compress(run_length_encoded)


huffman_decoded = huffman.decode_text(huffman_encoded)
run_length_decoded = reverse_run_length(run_length_encoded)

two_d = oneD2twoD(run_length_decoded)






############################################################################
# inverse discrete wavelet transform

def idwt(img,n):


	g0=np.array([1/2 , 1 , 1/2])

	g1=np.array([ 1/8 , -1/4 , 3/4 , -1/4 , 1/8 ])

	rows , columns = img.shape		


	######################
	x1 = np.zeros((img.shape[0],img.shape[1]))

	for column in range(columns//2):
		x1[:,column] = filter(upsample(img[0:img.shape[0]//2,column]*n[0]), g0)[:-len(g0)+1]


	for row in range(x1.shape[0]):
		x1[row,:] = filter(upsample(x1[row,:x1.shape[1]//2]), g0)[:-len(g0)+1]


	######################
	x2 = np.zeros((img.shape[0],img.shape[1]))

	for column in range(columns//2):
		x2[:,column] = filter(upsample(img[0:img.shape[0]//2,column+columns//2]*n[1]), g0)[:-len(g0)+1]


	for row in range(x2.shape[0]):
		x2[row,:] = filter(upsample(x2[row,:x2.shape[1]//2]), g1)[:-len(g1)+1]

	######################
	x3 = np.zeros((img.shape[0],img.shape[1]))

	for column in range(columns//2):
		x3[:,column] = filter(upsample(img[img.shape[0]//2:,column]*n[2]), g1)[:-len(g1)+1]


	for row in range(x3.shape[0]):
		x3[row,:] = filter(upsample(x3[row,:x3.shape[1]//2]), g0)[:-len(g0)+1]



	######################
	x4 = np.zeros((img.shape[0],img.shape[1]))

	for column in range(columns//2):
		x4[:,column] = filter(upsample(img[img.shape[0]//2:,column+columns//2]*n[3]), g1)[:-len(g1)+1]


	for row in range(x4.shape[0]):
		x4[row,:] = filter(upsample(x4[row,:x4.shape[1]//2]), g1)[:-len(g1)+1]



	return (x1 + x2 + x3 + x4)




# img[0:img.shape[0]//4,0:img.shape[1]//4] = idwt(img[0:img.shape[0]//4,0:img.shape[1]//4],[1,1,1,1])
# img[0:img.shape[0]//2,0:img.shape[1]//2] = idwt(img[0:img.shape[0]//2,0:img.shape[1]//2], [1,1,1,1])
# img = idwt(img,[1,1,1,1])

img = two_d

img[0:img.shape[0]//4,0:img.shape[1]//4] = idwt(img[0:img.shape[0]//4,0:img.shape[1]//4],[1,2,3,4])
img[0:img.shape[0]//2,0:img.shape[1]//2] = idwt(img[0:img.shape[0]//2,0:img.shape[1]//2], [1,2,3,4])
img = idwt(img,[1,2,3,4])


print(img.shape)
plt.imshow(img, cmap = "gray")





