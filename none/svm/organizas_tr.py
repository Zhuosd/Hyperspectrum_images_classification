# -*- encoding: utf-8 -*-
'''
@File    :   organizas_tr.py
@Time    :   2022/07/12 09:27:47
@Author  :   jolly 
@Version :   python3.7
@Contact :   jmlw8023@163.com
'''
# import packets

import numpy as np

def organizas_tr(vectTR):

    classes = np.setdiff1d(np.unique(vectTR[:]), 0)

    nx, ny = vectTR.shape
    vect_TR_3 = np.zeros(nx, ny, classes.max())

    for valor in classes:
        for i in range(nx):
            for j in range(ny):
                if vectTR[i, j] != valor:
                    vect_TR_3[i, j, valor] = 0
                else:
                    vect_TR_3[i, j, valor] = valor

    return vect_TR_3, classes