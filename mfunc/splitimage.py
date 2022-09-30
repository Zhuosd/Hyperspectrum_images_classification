import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import warnings
warnings.filterwarnings('ignore')
import math
    
def mlogistic(w, x):
    m = w.shape[1] + 1
    xw = w.T @ x
    # aux = math.exp(xw)
    aux1 = np.exp(xw)
    aux = np.exp(w.T @ x)
    ex = np.tile(1 + np.sum(aux, 0), (m-1, 1))
    p = aux / ex
    # last class
    p_s = (1 - np.sum(p, axis=0)).reshape(1, -1)

    k = np.row_stack((p, p_s))

    return k

#   splitimage(img, train, w, scale, sigma)
def splitimage(input_, z, w, scale, sigma): 
    x = input_

    d, n = x.shape
    z = z.astype(np.uint32)
    sum_pr = z**2
    nz = np.sum(sum_pr, axis=0)
    n1 = int(np.floor(n / 80))
    # p = np.array([])
    for i in range(1, 80):
        start = ((i-1) * n1)
        x1 = x[:,  start : n1 * i]
        #     x(:,1:n1) = [];
        nx1 = np.sum(x1 ** 2, axis=0)
        [X1, Z1]= np.meshgrid(nx1, nz)
        # del nx1

        dist1 = Z1 - 2 * z.T @ x1 + X1
        K1 = np.exp(-(dist1 / 2 / scale / sigma ** 2))
        d, nn = K1.shape
        K1 = np.row_stack((np.ones((nn)), K1))
      
        p1 = mlogistic(w,K1)
        if i == 1:
            p = p1
        # p = p1
        if i >= 2:
            p = np.concatenate((p, p1), axis=1)
            # print(f'i = {i}, p is no none !')
        #     x(:,1:n1) = [];
    
    x1 = x[:, (79 * n1) : n]
  
    nx1 = np.sum(x1 ** 2, axis=0)
    X1, Z1 = np.meshgrid(nx1, nz)
    dist1 = Z1 - 2 * np.transpose(z) @ x1 + X1
    K1 = np.exp(-(dist1 / 2 / scale / sigma ** 2))
    d, nnn = K1.shape
    K1 = np.row_stack((np.ones((nnn)), K1))
    # K1 = np.array([[np.ones((1,n - 79 * n1))],[K1]])
    p1 = mlogistic(w, K1)
    p = np.concatenate((p, p1), axis=1)

    return p

