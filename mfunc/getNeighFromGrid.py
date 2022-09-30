## This function returns the ID of each node in a given grid and
## number and list of its neighbors

import numpy as np

def sub2ind(M, col, row):
    # row, col = row-1, col-1
    # print('row = %d, col = %d' % (row, col))
    # print("sub2ind {}".format(M.shape))
    num_rows = M.shape[0]       # 获取矩阵M的行数
    # print(num_rows)
    if int(row) == 0 and int(col) == 0:
        return 1
    return int(num_rows * (row-1) + col)


def getNeighFromGrid(rows, cols): 
    # if cliques == 1
    maxNumN = 4
    # M = np.ones([rows,cols])
    M = np.ones([596, 601])
    nList = np.zeros((rows * cols, maxNumN))
    # nList = np.zeros((596 * 601, maxNumN))
    # numN = np.zeros((1197), dtype=int)
    numN = np.zeros((rows * cols +1), dtype=int)

    for j in np.arange(1, cols+1):
        for i in np.arange(1, rows+1): 
    # i, j = rows, cols
            curID = sub2ind(M, i, j)
            # print('curID -- {}'.format(curID))
            # print('curID type -- {}'.format(type(curID)))
            # print(' # ' * 20)
            # print('curID -- {}'.format(type(curID)))
            # numN[curID] = 0
            # numN = np.zeros((curID), dtype=int)
            # print('id(numN) -- {}'.format(id(numN)))
            # print(numN)
            if (i - 1) > 0:
                numN[curID-1] = numN[curID-1] + 1
                # print( numN[-1])
                # print( numN.size)
                # print(' - ' * 15)
                # print(nList.shape)
                nList[curID-1, numN[curID-1]-1] = sub2ind(M, i - 1, j)
                # print(nList[curID-1, -1])
            if (j - 1) > 0:
                numN[curID-1] = numN[curID-1] + 1
                nList[curID-1, numN[curID-1]-1] = sub2ind(M, i, j - 1)
            if (i + 1) <= rows:
                # print(numN[curID])
                numN[curID-1] = numN[curID-1] + 1
                nList[curID-1, numN[curID-1]-1] = sub2ind(M, i + 1, j)
            if (j + 1) <= cols:
                numN[curID-1] = numN[curID-1] + 1
                # print(' - ' * 20)
                # print(f'numN.shape = {numN.shape}')
                # print(f'nList.shape = {nList.shape}')
                # print(f'[curID-1] = {curID-1} --')
                # print(f'numN[curID-1] = {numN[curID-1]} --')
                # print(f'numN[curID-1] -1 = {numN[curID-1]-1} --')
                # print(f'sub2ind(M, i, j + 1) = {sub2ind(M, i, j + 1)} --')
                nList[curID-1, numN[curID-1]-1] = sub2ind(M, i, j + 1)
            # RT = numN
    numN = np.delete(numN, -1)
    return numN, nList




if __name__ == '__main__':

    rows, cols = 596, 601
    # rows, cols = 8, 2

    numN, nList = getNeighFromGrid(rows, cols)

    print(numN)
    print(numN.shape)
    # print(nList)
    # print(nList[:, 0])
    print(nList[593:612, :])
    print(' - ' * 20)
    print(nList[357590:357612, :])
    print(' - ' * 20)
    print(nList)
    print(nList.shape)
    print(' - ' * 20)
    print(numN)
    print(numN[1783:1793])
    print(numN.shape)
    # i, j = rows, cols

    # M = np.ones([596, 601])
    # print(M.shape)
    # curID = sub2ind(M, i, j)
    # print('curID -- {}'.format(curID))
    # print(' # ' * 20)
