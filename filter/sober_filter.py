import numpy as np
import cv2

with open('lena.raw') as raw:
    XSIZE, YSIZE = 512, 512
    img = np.fromfile(raw, dtype=np.uint8, count=XSIZE*YSIZE)
    img.shape = (img.size // XSIZE, YSIZE)

# img 배열 주변에 0으로 둘러싸기
zero_row = np.zeros((1, XSIZE), dtype=int)
zero_col = np.zeros((YSIZE+2, 1), dtype=int)
convol = np.column_stack([zero_col, np.vstack([zero_row, img, zero_row]), zero_col])

#필요한 배열 선언
sobel_mask_output = np.empty((XSIZE, YSIZE), dtype=int)
mask = np.empty((3, 3), dtype=int)

#sobel mask
sobel_mask_X = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_mask_Y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

max_v, min_v = None, None

# convol[i][j]에 맞는 mask 만들기
def find_mask(x, y):
    for j in range(3):
        for i in range(3):
            mask[i][j] = convol[x+i-1][y+j-1]

# ∆f ≈ |Gx| + |Gy|
# Gx = mask*sobel_mask_X의 합
# Gy = mask*sobel_mask_Y의 합
def sobel_operator():
    value_X = (mask*sobel_mask_X).sum()
    value_Y = (mask*sobel_mask_Y).sum()
    value = (abs(value_X) + abs(value_Y))/2
    if value > 255:
        value = 255
    return value

def detect_edge():
    for j in range(1, YSIZE+1):
        for i in range(1, XSIZE+1):
            find_mask(i, j)
            sobel_mask_output[i-1][j-1] = sobel_operator()

if __name__ == "__main__":
    detect_edge()
    sobel_mask_output.astype('uint8').tofile('sobel_mask.raw')

    with open('sobel_mask.raw') as raw:
        XSIZE, YSIZE = 512, 512
        sobel = np.fromfile(raw, dtype=np.uint8, count=XSIZE * YSIZE)
        sobel.shape = (sobel.size // XSIZE, YSIZE)
    cv2.imshow('Sobel mask', sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
