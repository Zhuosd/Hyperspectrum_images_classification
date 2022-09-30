import numpy as np
from sklearn.model_selection import train_test_split

def train_test_random_new(y ,n): 
    # function to ramdonly select training samples and testing samples from the
    K = int(np.amax(y[1:]))     # k is 8
    # print('k is %d' % K)
    # print(type(y))      #  <class 'numpy.ndarray'>
    # print((y).shape)    # y.shape is (2, 23821)
    # K = int(np.amax(trainall[1:, :]))   # K is 8
    temp = []
    indexes = np.matrix([])
    for i in range(K):
        nu = i + 1 
        index1 = np.matrix((y[1, :] == nu).nonzero())

        # print((index1))
        # print((index1).shape)
        # print(' index1 '*10)
        len = index1.size
        per_index1 = np.random.permutation(range(len))  #  打乱len 个元素
        # print(per_index1)
        if len > n:
            per = per_index1[:n]        # 取出 乱序per_index1 的前 n个值， 此处n=5
            M = np.transpose(index1)    # 转置为 1 * index1.size 维度
            for num in range(int(n)):
                # print((M[per[num]]).shape)        # shape is (1, 1)
                # print(type(M[per[num]]))        # type is class 'numpy.matrix'
                temp.append(M[per[num]][0, 0])    #  取出（1,1）中矩阵的值M[0, 0]

            # other methods
            # num = int(n * K)
            # x_train, x_test = train_test_split(per_index1, train_size=num)
            # print((x_train))
            # print((x_train).shape)
            # print((x_test).shape)

        else:
            print('-'*30)
            n0 = np.ceil((index1.size) / 2)
    #         # indexes = np.array([indexes,[np.transpose(index1(per_index1[:n0]))]])
            per = per_index1[:n0]
            M = np.transpose(index1)    # 转置为 1 * index1.size 维度
            # print(per)
            for num in range(int(n0)):
                # print((M[per[num]]).shape)        # shape is (1, 1)
                # print(type(M[per[num]]))        # type is class 'numpy.matrix'
                temp.append(M[per[num]][0, 0])    #
            # x_train, x_test = train_test_split(per_index1, train_size=n0)

    # temp = np.matrix(temp)
    # print((temp))   # temp type is list 
    x_train = np.array(temp)       # 转为  len(index1) * 1 维度矩阵， 此处 为40 * 1

    return x_train

