from ast import arg
import numpy as np
# import pywt
import scipy.io
import scipy

# lorsal(Ktrain, train_set[1, :], lambda_, 0.01 * lambda_, MMiter, 0)
def lorsal_test(*args): 
    in_arg_len = len(args)
    print(f'in_arg_len =  {in_arg_len}')
    if in_arg_len > 5:
        print('go in ' * 5)
        x, y, lambda_, beta, MMiter, w0 = args[0], args[1], args[2], args[3], args[4], args[5]
    # #  Sparse Multinomial Logistic Regression
    # x = np.dtype('uint32').type(x)
    # y = np.dtype('uint32').type(y)

    #  Implements a block Gauss Seidel  algorithm for fast solution of
    #  the SMLR introduced in Krishnapuram et. al, IEEE TPMI, 2005 (eq. 12)
        # -- Input Parameters ----------------     
        # x ->      training set (each column represent a sample).
    #           x can be samples or functions of samples, i.e.,
    #           kernels, basis, etc.
    # y ->      class (1,2,...,m)
    # lambda -> sparsness parameter
    # MMiter -> Number of iterations (default 100)
    # Author: Jose M. Bioucas-Dias, Fev. 2006
    # Bregman weight


    path_mat = r'E:\source\code\matlab\Demo_DRF_EM_Zaoyuan\Ktrain.mat'
    Ktrain_dicts = scipy.io.loadmat(path_mat)
    Ktrain = Ktrain_dicts['Ktrain']

    x = Ktrain

    # beta = 0.01 * lambda_;
    # Bregman iterations
    BR_iter = 1
    # Block iterations
    Bloc_iters = 1
    if in_arg_len < 5:
        MMiter = 200
    
    #[d - space dimension, n-number of samples]
    d, n = x.shape
    # print(f'd = {d} , n = {n} ')
    # if (size(y,1) ~= 1) | (size(y,2) ~= n)
    #     error('Input vector y is not of size [1,#d]',n)
    # end
    # number of classes
    m = int(y.max(0))
    # print(f'm = {m} ')
    # auxiliar matrix to compute a bound for the logistic hessian
    U = -1/2 * (np.eye(m - 1) - np.ones((m - 1, m - 1)) / m)
    
    # convert y into binary information
    Y = np.zeros((m,n))
    # print(Y)
    # print(Y.shape)
    
    for i in range(n):
        t_y = int(y[i]) -1
        Y[t_y, i] = 1
    
    Y = Y[:m-1, :]
    Rx = x @ x.T

    print(f'in_arg_len =  {in_arg_len} ')
    if in_arg_len <= 6:
        alpha = 1e-05
        w_t = (Rx + alpha * np.eye(d))
        w_tt = (x @ (10 * Y.T - 5 * np.ones(Y.T.shape)))
        w = np.linalg.solve(w_t, w_tt)

    pu = r'E:\\source\\code\\matlab\\Demo_DRF_EM_Zaoyuan\\Uu.mat'
    Uu_ = scipy.io.loadmat(pu)
    Uu = Uu_['Uu']
    path_mat = r'E:\source\code\matlab\Demo_DRF_EM_Zaoyuan\Du.mat'
    Du_ = scipy.io.loadmat(path_mat)
    Du = Du_['Du']

    path_mat = r'E:\source\code\matlab\Demo_DRF_EM_Zaoyuan\Dr.mat'
    Ktrain_dicts = scipy.io.loadmat(path_mat)
    Dr = Ktrain_dicts['Dr']

    pr = r'E:\source\code\matlab\Demo_DRF_EM_Zaoyuan\Ur.mat'
    Ur_ = scipy.io.loadmat(pr)
    Ur = Ur_['Ur']
    Dr = Dr.real
    Ur = Ur.real

    print(' + ' * 15)

    ex_dr = np.tile(np.diag(Dr), (m - 1, 1)).T # reshape is (181, 1)
    ex_d = beta * np.ones((d, m - 1))
    S = 1.0 / ((ex_dr @ Du) - ex_d)

    # -----------
    # Bregman iterative scheme to compute w
    # initialize v (to impose the constraint w=v)
    v = w
    # # initialize the Bregman vector b
    b = np.zeros(v.shape)
    # print(f' b.shape = {b.shape}')
    # MM iterations
    L = []
    L.clear()
    # L = np.zeros(125)
    for i in np.arange(MMiter):
        # for i in np.arange(MMiter):
        print('\n i = {}'.format(i))
        # #compute the  multinomial distributions (one per sample)
        # print('w.T.dot(x)' * 10)
        val =  (w.T @ x)
        aux1 =  np.exp(val)
        # aux1[np.isneginf(aux1)] = 0
        # aux1[np.isposinf(aux1)] = 10
        # if val >= 0:
        aux2 =  (1 + np.sum(aux1, axis=0))
        # aux2[np.isneginf(aux2)] = 0
        # aux2[np.isposinf(aux2)] = 100
        # print(aux2.shape)
        p_dev =  np.tile(aux2.reshape(1, -1),(m - 1, 1))
        p = aux1 / p_dev
        # print(f'p.shape = {p.shape}')
        # compute log-likelihood
        
        # sum_a = np.sum((Y * val), axis=0) 
        sum_a = (Y * val).sum(axis=0)
        # L_ = np.sum(np.sum(np.multiply(Y, val), axis=0) - np.log(aux2)) - lambda_ * np.sum(np.abs(w)) + lambda_ * sum(np.abs(w[0,:]))
        L_ = np.sum(sum_a - np.log(aux2)) - lambda_ * np.sum(np.abs(w)) + lambda_ * np.sum(np.abs(w[0,:]))
        # print(L[i])
        L.append(L_)

        # db(i) = norm(w);
        # bd(i) = norm(w-v);
        # compute derivative of the multinomial logistic regression at w2
        # g = x*(Y-p)';
        # compute derivative
        # dg = Rx.dot(w.dot(U.T)) - x.dot((Y - p).T)
        # dg = np.dot(Rx.dot(w), U.T) - np.dot(x, (Y-p).T)
        dg = Rx @ w @ U.T - x @ (Y-p).T
        # print('dg.shape = {}'.format(dg.shape))
        # Bregman iterations
        for k in range((BR_iter)):
        # for k in np.arange(BR_iter):
        #     for j in np.arange(Bloc_iters):
            for j in range((Bloc_iters)):
                # update w
                print(f'===k = {k} ========j  = {j}=========================')
                z = dg - beta * (v + b)
                # wwww = np.dot(Ur.T.dot(z), Uu)
                w = S * (Ur.T @ z @ Uu)

                # w = np.multiply(S, np.multiply(Ur.T,(z.dot(Uu))))
                # w = np.dot(Ur.dot(w), Uu.T)
                w = Ur @ w @ Uu.T
                # update v
                # v = wthresh(w - b,'s',lambda_ / beta)
                v = pywt.threshold((w - b), lambda_ / beta, 'soft')
            # Bregman factor
            b = b - (w - v)
            # norm(b)


        beta *=  1.05
        # sr = (np.tile(Dr, (m - 1, 1))).T
        # sr = np.tile(Dr.reshape(-1, 1), (1, m-1))
        sr = np.tile(np.diag(Dr), (m-1, 1)).T
        so = beta * np.ones((d, m - 1))
        # S = 1.0 / (np.multiply(sr, Du) - so)
        # S = 1.0 / (sr @ np.diag(Du) - so)
        S = 1.0 / (sr @ (Du) - so)
        # S = 1.0 / (np.matmul((np.tile(Dr, (m - 1, 1))).T, Dr) - beta * np.ones((d, m - 1)))
        # print(S)
        # print(S.shape)
        # print(' S ' * 15)
        
        # p(m,:) = 1-sum(p,1);

    # print(L)
    return v, np.array(L)