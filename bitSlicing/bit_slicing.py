import numpy as np
fp = open('./bitSlicing/lena_water_mark.raw', 'rb')
XSIZE, YSIZE = 512, 512
img = np.fromfile(fp,dtype = np.uint8, count =XSIZE*YSIZE )
img.shape = (img.size // XSIZE,XSIZE)
binary_img = []

#Iterate over each pixel and change pixel value to binary using np.binary)repr() and store it in a list named binary_img
for i in range(XSIZE):
    for j in range(YSIZE):
        binary_img.append(np.binary_repr(img[i][j], width= 8))

#Multiply each pixel by 15, rearray, name each bit plane and save as file
for i in range(8):
    bit_plane = (np.array([int(j[i]) for j in binary_img], dtype = np.uint8)*255).reshape(XSIZE,YSIZE)
    image_name = 'bitplane_'+ str(i) + '.raw'
    bit_plane.astype('int8').tofile(image_name)