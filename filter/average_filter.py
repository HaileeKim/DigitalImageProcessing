import numpy as np
fp = open('output_lena_GN_std_8.raw', 'rb')
XSIZE, YSIZE = 512, 512
img = np.fromfile(fp, dtype=np.uint8, count=XSIZE*YSIZE)
img.shape = (img.size // XSIZE, XSIZE)

#img 배열 주변에 0으로 둘러싸기
zero_row = np.zeros((1, XSIZE), dtype=int)
zero_col =np.zeros((YSIZE+2, 1), dtype=int)
convol = np.column_stack([zero_col, np.vstack([zero_row, img, zero_row]), zero_col])
convol_mask = np.zeros((3,3), dtype=int)
avg_mask = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
weight_mask = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])


avg_filter = np.empty((XSIZE, YSIZE))
weight_filter = np.empty((XSIZE, YSIZE))

#mask 를 찾고 평균/가중치에 해당하는 값을 곱한 것의 합을 리
def mask_opeartor(x, y):
    for i in range(3):
        for j in range(3):
            convol_mask[i][j] = convol[x+i-1][y+j-1]
    avg = sum(sum(avg_mask*convol_mask))
    weight = sum(sum(weight_mask*convol_mask))
    return avg, weight


for i in range(1, XSIZE+1):
    for j in range(1, YSIZE+1):
        avg_filter[i-1][j-1], weight_filter[i-1][j-1] = mask_opeartor(i,j)
avg_filter.astype('int8').tofile('avg_filter.raw')
weight_filter.astype('int8').tofile('weighted_filter.raw')