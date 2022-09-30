# -*- encoding: utf-8 -*-
'''
@File    :   aux_ordenar_v4.py
@Time    :   2022/07/12 10:06:31
@Author  :   jolly 
@Version :   python3.7
@Contact :   jmlw8023@163.com
'''

# import packets
# import os
from numpy import array
import numpy as np
import organizas_tr



def aux_ordenar_v4(vect_TR,vectProb,no_lines,no_col):
    '''
    Inertiment function
    :param vect_TR:
    :param vectProb:
    :param no_lines:
    :param no_col:
    :return:
    '''
    vect_classes = []
    vect_TR_3, classes = organizas_tr(vect_TR)
    x, y, b = vect_TR_3.shape
    vect_TR_3 = np.mat(vect_TR_3)
    vect_TR_3_2 = vect_TR_3.reshape(x*y, b)
    for i in classes:
        sum_vect = vect_TR_3_2.sum(axis=0)
        vect_classes = array(vect_classes, sum_vect)
    
    # p_tr= orden de las clases
    # valor, p_tr = sorted(vect_classes, reverse=True)
    valor, p_tr = np.sort(vect_classes)[::-1]

    # Leer el vector de probabilidad (valor, posicion) (21025*16)
    ns, p = vectProb.shape 

    # nuevas variables
    M = []

