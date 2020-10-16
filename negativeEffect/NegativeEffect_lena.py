import numpy as np
fp = open('lena.raw', 'rb')
XSIZE, YSIZE = 512, 512
img = np.fromfile(fp,dtype = np.uint8, count =XSIZE*YSIZE )
img.shape = (img.size // XSIZE,XSIZE)
for i in range(XSIZE):
    for j in range(YSIZE):
        img[i][j] = 255 - img[i][j]
img.astype('int8').tofile('newImage.raw')
