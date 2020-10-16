import numpy as np
fp = open('output_lena_GN_std_8.raw', 'rb')
XSIZE, YSIZE = 512, 512
img = np.fromfile(fp, dtype=np.uint8, count=XSIZE*YSIZE)
img.shape = (img.size // XSIZE,XSIZE)

#img 배열 주변에 0으로 둘러싸기
zero_row = np.zeros((1, XSIZE), dtype=int)
zero_col = np.zeros((YSIZE+2, 1), dtype=int)
convol = np.column_stack([zero_col, np.vstack([zero_row, img, zero_row]), zero_col])

median_filter = np.empty((XSIZE, YSIZE))
mask = np.empty((9), dtype=int)

#convol[i][j]에 맞는 mask 만들기
def find_mask(x, y):
    count = 0
    a = x-1
    b = y-1
    for i in range(9):
        mask[i] = convol[a][b]
        a += 1
        count += 1
        if count % 3 == 0:
            a = x-1
            b += 1
    return mask

for i in range(1, XSIZE+1):
    for j in range(1, YSIZE+1):
        median_filter[i-1][j-1] = sorted(find_mask(i, j))[4] #maks 크기 순으로 정렬 후 가운데 값 추출

median_filter.astype('int8').tofile('median_filter.raw')