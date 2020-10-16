import numpy as np
import cv2

with open('white_lena.raw') as raw:
    XSIZE, YSIZE = 512, 512
    img = np.fromfile(raw, dtype=np.uint8, count=XSIZE*YSIZE)
    img.shape = (img.size // XSIZE, YSIZE)

output = np.zeros((XSIZE, YSIZE), dtype=int)


def normalized(img):
    n = XSIZE * YSIZE
    norm = np.zeros(XSIZE + 1, dtype=int)
    for j in range(YSIZE):
        for i in range(XSIZE):
            norm[img[i][j]] += 1
    norm = norm / n
    return norm


def histogram_equalization():
    s_k = []
    norm = normalized(img)
    norm_sum = 0
    for r_k in range(256):
        norm_sum += norm[r_k]*255
        s_k.append(norm_sum)
    for j in range(YSIZE):
        for i in range(XSIZE):
            output[i][j] = s_k[img[i][j]]

if __name__ == "__main__":
    histogram_equalization()

    output.astype('uint8').tofile('equalization.raw')

    with open('equalization.raw') as raw:
        XSIZE, YSIZE = 512, 512
        equal = np.fromfile(raw, dtype=np.uint8, count=XSIZE * YSIZE)
        equal.shape = (equal.size // XSIZE, YSIZE)

    # cv2.imshow('lena', img)
    cv2.imshow('Sobel mask', equal)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
