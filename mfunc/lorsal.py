import os
import numpy as np
import pywt
import torch
import scipy.io
from scipy import linalg
from sympy import *
import pandas as pd

# torch
# numpy
# scipy
def lorsal(*args):
    in_arg_len = len(args)
    if in_arg_len > 5:
        print('go in ' * 5)
        x, y, lambda_, beta, MMiter, w0 = args[0], args[1], args[2], args[3], args[4], args[5]

    input_data = True

    root_path = r"D:/PostGraduate_Guangzhuo/2022-研三/论文复现/DRF_EM_python--20220921/libmat/val/"
    if input_data:
        path_mat = os.path.join(root_path, 'Ktrain.mat') 
        Ktrain_dicts = scipy.io.loadmat(path_mat)
        Ktrain = Ktrain_dicts['Ktrain']
        x = Ktrain

    BR_iter = 1
    Bloc_iters = 1
    if in_arg_len < 5:
        MMiter = 200

    d, n = x.shape
    m = int(y.max(0))
    U = -1/2 * (np.eye(m - 1) - np.ones((m - 1, m - 1)) / m)

    Y = np.zeros((m,n))

    for i in range(n):
        t_y = int(y[i]) -1
        # print(t_y)
        Y[t_y, i] = 1
    
    Y = Y[:m-1, :]

    Rx = x @ x.T
    print(f'in_arg_len =  {in_arg_len} ')
    if in_arg_len <= 6:
        alpha = 1e-05
        w_t = (Rx + alpha * np.eye(d))
        w_tt = (x @ (10 * Y.T - 5 * np.ones(Y.T.shape)))
        w = np.linalg.solve(w_t, w_tt)

    matlab_flag = False # False / True

    if matlab_flag:
        pu = os.path.join(root_path, 'Uu.mat') 
        Uu_ = scipy.io.loadmat(pu)
        Uu = Uu_['Uu']
        path_mat = os.path.join(root_path, 'Du.mat')
        Du_ = scipy.io.loadmat(path_mat)
        Du = Du_['Du']
    else:
        #
        U_test = Matrix(U)
        DU_test = U_test.eigenvals()
        Uu_test = U_test.eigenvects()

        Du, Uu = np.linalg.eigh(U) # np.linalg.eig(U)
        Du, Uu = np.around(Du, 6), np.around(Uu, 6)
        ind = np.argsort(Du)      # 返回值进行排序，MATLAB会自动排序
        print(ind)
        Du = Du[ind]
        Uu = Uu[:, ind]
        Du = np.real(Du)
        Uu = np.real(Uu)

    if matlab_flag:
        path_mat = os.path.join(root_path, 'Dr.mat')
        Ktrain_dicts = scipy.io.loadmat(path_mat)
        Dr = Ktrain_dicts['Dr']

        pr = os.path.join(root_path, 'Ur.mat')
        Ur_ = scipy.io.loadmat(pr)
        Ur = Ur_['Ur']

    else:
        Dr, Ur = linalg.eig(Rx)
        # Dr, Ur = np.linalg.eig(Rx)
        Dr, Ur = np.around(Dr, 6), np.around(Ur, 6)
        indr = np.argsort(Dr)    # 返回值进行排序，MATLAB会自动排序
        # print(indr)
        Dr = Dr[indr]
        Ur = Ur[:, indr]
        Dr = Dr.real
        Ur = Ur.real

        # Dr1, Ur1 = torch.linalg.eig(torch.from_numpy(Rx))
        # indr = np.argsort(Dr1.numpy().real)    # 返回值进行排序，MATLAB会自动排序
        # dr = np.around(Dr1.numpy().real, 6)
        # ur = np.around(Ur1.numpy().real, 6)
        # Dr = dr[indr]
        # Ur = ur[:, indr]

    print("Uu", Uu)
    print("Ur", Ur)
    # Ur[:, :180] = 0
    # Ur[:, 3:181] = 0

    if matlab_flag:
        ex_dr = np.tile(np.diag(Dr), (m - 1, 1)).T # load  is (181, 1)
        ex_d = beta * np.ones((d, m - 1))
        S = 1.0 / ((ex_dr @ (Du)) - ex_d)   # load
    else:
        ex_dr = np.tile(Dr, (m - 1, 1)).T # reshape is (181, 1)
        ex_d = beta * np.ones((d, m - 1))
        S = np.around((1.0 / ((ex_dr @ np.diag(Du)) - ex_d)), 6)
    v = w
    b = np.zeros(v.shape)
    L = []
    L.clear()
    for i in np.arange(MMiter):
        w_T_x = w.T @ x
        val =  np.around((w.T @ x), 6)
        aux1 =  np.exp(val)
        aux2 =  (1 + np.sum(aux1, axis=0))
        p_dev =  np.tile(aux2.reshape(1, -1),(m - 1, 1))
        p = aux1 / p_dev
        sum_a = (Y * val).sum(axis=0)
        sum_1 = lambda_ * np.sum(np.abs(w))
        sum_2 = lambda_ * np.sum(np.abs(w[0,:]))
        sum_all = np.sum(sum_a - np.log(aux2))
        L_ = sum_all - sum_1 + sum_2
        L.append(L_)

        dg = Rx @ w @ U.T - x @ (Y-p).T
        for k in range((BR_iter)):
            for j in range((Bloc_iters)):
                z = dg - beta * (v + b)
                w = S * (Ur.T @ z @ Uu)
                w = Ur @ w @ Uu.T
                v = pywt.threshold((w - b), lambda_ / beta, 'soft')
            b = b - (w - v)
        beta *=  1.05
        so = beta * np.ones((d, m - 1))
        if matlab_flag:
            sr = np.tile(np.diag(Dr), (m-1, 1)).T   # load
            S = 1.0 / (sr @ Du - so)    # load
        else:
            sr = np.tile(Dr, (m-1, 1)).T
            S = 1.0 / (sr @ np.diag(Du) - so)

    return v, np.array(L)


