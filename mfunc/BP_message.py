from array import array
import time
import numpy as np
    
def BP_message(*args): 
    in_arg_len = len(args)
    if in_arg_len <= 5:
        phi, psi, nList, trainold, BPiter = np.round(args[0], 8), args[1], \
                                            np.round(args[2], 8), np.round(args[3], 8), \
                                            np.round(args[4], 8)
    #  For any comments contact the authors
    tag = False
    start_time = time.process_time()
    if tag:
        print('-- start time----')
    sz = nList.shape
    sz_train = trainold.shape
    K = psi.shape[1]
    
    if in_arg_len < 6:
        flg = 1
    
    if in_arg_len < 5:
        BPiter = 10
    
    ##  intialize the msg
    # msg  sz*m*K  n is the number of samples, m is the number of neighbors, K
    # is the number of classes
    msg = np.zeros((sz[0], sz[1], K))
    for i in np.arange(sz[0]):
        xList = np.where(nList[i, :] != 0)[0].size  # 返回整数
        for j in np.arange(xList):
            msg0 = phi[i, :] @ psi
            msg0_temp = np.round(np.sum(msg0, axis=0), 6)
            msg0_temp = np.tile(msg0_temp, (1,K))
            msg0 = msg0 / msg0_temp
            msg[i, j, :] = msg0
    if tag:
        print('finish first cycle!! ')
        end_time = time.process_time()
        print(f'using {end_time - start_time} seconds!!')
    ## assign the training set
    
    if flg == 1:
        psi2_beta = np.round(psi[0,0], 6)
        psi2 = psi.copy()
        for k in np.arange(K):
            psi2[k,k] = psi[0, 1]
        for i in np.arange(sz_train[1]):
            psi2[int(trainold[1,i])-1, int(trainold[1,i])-1] = psi2_beta

            xList = np.where(nList[int(trainold[0, i])-1, :] != 0)[0].size  # 返回整数
            for j in np.arange(xList):
                msg0 = phi[int(trainold[0, i])-1, :] @ psi2
                msg0_temp = np.round(np.sum(msg0, axis=0), 4)
                msg0_temp = np.tile(msg0_temp, (1, K))
                msg0 = np.round((msg0 / msg0_temp), 8)
                msg[int(trainold[0,i])-1, j, :] = msg0
    
    if tag:
        print('second cycle!! ')
        end_time = time.process_time()
        print(f'using {end_time - start_time} seconds!!')
    ## update message
    
    msg_temp0 = msg
    for iter in np.arange(BPiter):
        for i in np.arange(sz[0]):
            # xList = nList[i, :]
            # # xList[xList == 0] = []
            # flag = np.where(xList == 0)[0]
            # # print(flag)
            # xList = np.delete(xList, flag)
            xList = np.delete(nList[i, :], np.where(nList[i, :] == 0)[0])
            # xList = np.where(nList[i, :] != 0)[0].size  # 返回整数
            # for j in np.arange(xList):
            for j in np.arange(len(xList)):
                yList = xList
                # yList[j] = []
                yList = np.delete(yList, j)
                msg_temp = np.multiply(np.tile(np.round(phi[i, :], 4),(K, 1)), psi)   # np.multiply 与 * 一致
                msg_bp = np.array([])
                for n_prod in np.arange(len(yList)):
                    yList2 = nList[int(yList[n_prod])-1, :]
                    yj_ind1 = np.int8(yList2 == i+1)
                    index = np.where(yj_ind1 == 0)[0]
                    yj_ind = np.delete(yj_ind1, index)
                    # msg_bp = np.squeeze(msg_temp0[int(yList[n_prod])-1, yj_ind,:]).reshape(-1, 1)
                    msg_bp = np.squeeze(msg_temp0[int(yList[n_prod])-1, yj_ind,:])
                msg0 = msg_temp @ np.prod(msg_bp.reshape(-1, 1), axis=1)
                # msg0 = msg_temp @ msg_bp
                msg0 = np.transpose(msg0)
                msg0_temp = np.sum(msg0, axis=0)
                msg0_temp = np.tile(msg0_temp, (1,K))
                msg0 = msg0 / msg0_temp
                msg[i,j,:] = msg0

        ## assign the training set
        if flg == 1:
            psi2 = psi.copy()
            for k in np.arange(K):
                psi2[k,k] = psi[0, 1]
            for i in np.arange(sz_train[1]):
                psi2[int(trainold[1,i])-1, int(trainold[1,i])-1] = psi2_beta
                xList = nList[int(trainold[0, i])-1, :]
                # # xList[xList == 0] = []
                flag = np.where(xList == 0)[0]
                xList = np.delete(xList, flag)
                for j in np.arange(len(xList)):
                # xList = np.where(nList[int(trainold[0, i])-1, :] != 0)[0].size  # 返回整数
                # for j in np.arange(xList):
                    yList = xList
                    # yList[j] = []
                    yList = np.delete(yList, j)
                    # print("np.tile(phi[trainold[0, i]-1, :], (K,1))", np.tile(phi[trainold[0, i]-1, :], (K,1)))
                    # print("psi2", psi2)
                    msg_temp = np.multiply(np.tile(phi[int(trainold[0, i]) - 1, :], (K, 1)), psi2)
                    # msg_temp = np.multiply(np.tile(phi[trainold[0, i]-1, :], (K,1)), psi2)
                    msg_bp = np.array([])
                    for n_prod in np.arange(len(yList)):
                        yList2 = nList[int(yList[n_prod])-1, :]
                        # yj_ind = yList2 == trainold[0, i]
                        yj_ind1 = np.int8(yList2 == trainold[0, i])
                        ind = np.where(yj_ind1 == 0)[0]
                        yj_ind = np.delete(yj_ind1, ind)

                        msg_bp = np.squeeze(msg_temp0[int(yList[n_prod])-1, yj_ind, :])
                    msg0 = msg_temp @ np.prod(msg_bp.reshape(-1, 1), axis=1)
                    msg0 = np.transpose(msg0)
                    msg0_temp = np.sum(msg0, axis=0)
                    msg0_temp = np.tile(msg0_temp, (1, K))
                    msg0 = msg0 / msg0_temp
                    msg[int(trainold[0,i])-1, j, :] = msg0
        msg_temp0 = msg
    
    if tag:
        print('third cycle!! ')
        end_time = time.process_time()
        print(f'using {end_time - start_time} seconds!!')
    ##  compute Beliefs
    belief = np.zeros((phi.shape[0], phi.shape[1]))
    for i in np.arange(sz[0]):
        xList = nList[i, :]
        flag = np.where(xList == 0)[0]
        # print(flag)
        xList = np.delete(xList, flag)
        # xList[xList == 0] = []
        yList = xList
        msg_bp = np.array([])
        for n_prod in np.arange(len(yList)):
            yList2 = nList[int(yList[n_prod])-1, :]
            yj_ind = np.int8(yList2 == i+1)
            index = np.where(yj_ind == 0)[0]
            yj_ind = np.delete(yj_ind, index)
            msg_bp = np.squeeze(msg_temp0[int(yList[n_prod])-1, yj_ind, :])
        belief[i,:] = np.multiply(phi[i,:], np.transpose(np.prod(msg_bp.reshape(-1, 1), axis=1)))
    
    if tag:
        print('fourth cycle!! ')
        end_time = time.process_time()
        print(f'using {end_time - start_time} seconds!!')
    belief = np.transpose(belief)
    belief_temp = np.sum(belief, axis=0)
    belief_temp = np.tile(belief_temp,(K,1))
    belief = belief / belief_temp
    return belief

