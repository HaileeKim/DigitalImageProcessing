import numpy as np
fp = open('lena.raw', 'rb')
XSIZE, YSIZE = 512, 512
img = np.fromfile(fp, dtype=np.uint8, count=XSIZE*YSIZE)
img.shape = (img.size // XSIZE, YSIZE)

# img 배열 주변에 0으로 둘러싸기
zero_row = np.zeros((1, XSIZE), dtype=int)
zero_col = np.zeros((YSIZE+2, 1), dtype=int)
convol = np.column_stack([zero_col, np.vstack([zero_row, img, zero_row]), zero_col])

#필요한 배열 선언
laplacian_filter_center4 = np.empty((XSIZE, YSIZE), dtype=int)
laplacian_filter_center8 = np.empty((XSIZE, YSIZE), dtype=int)
mask = np.empty((3, 3), dtype=int)
std = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
# 4방향 가중치
weight_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
# 8방향 가중치
weight_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

# convol[i][j]에 맞는 mask 만들기
def find_mask(x, y):
    for i in range(3):
        for j in range(3):
            mask[i][j] = convol[x+i-1][y+j-1]
# 라플라시안 마스크를 이용해 sharpening 하기
# g(x,y) = f(x,y) - ∆^2f(x,y), center of mask < 0
# g(x,y) = f(x,y) + ∆^2f(x,y), center of mask > 0
def laplacian_operator(weight):
    value = 0
    if weight == 4:
        if weight_4[1][1] < 0:
            value = sum(sum((std - weight_4)*mask))
        else:
            value = sum(sum((std + weight_4) * mask))
    elif weight == 8:
        if weight_8[1][1] < 0:
            value = sum(sum((std - weight_8)*mask))
        else:
            value = sum(sum((std + weight_8)*mask))
    # overflow & underflow 막기
    if value < 0:
        value = 0
    elif value > 255:
        value = 255
    return value

# main문
if __name__ == "__main__":
    for j in range(1, YSIZE + 1):
        for i in range(1, XSIZE + 1):
            find_mask(i, j)
            laplacian_filter_center4[i - 1][j - 1] = laplacian_operator(4)
            laplacian_filter_center8[i - 1][j - 1] = laplacian_operator(8)
    laplacian_filter_center4.astype('int8').tofile('laplacian_filter_center4.raw')
    laplacian_filter_center8.astype('int8').tofile('laplacian_filter_center8.raw')
