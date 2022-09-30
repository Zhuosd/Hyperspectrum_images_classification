## Demo: Applying DRF-EM to the AVIRIS Indian Pines scene
## RBF with with all bands
## BP for posterior probability estimation
import os
import time
import scipy
import scipy.io
import scipy.io as io
import numpy as np
from mfunc.splitimage import splitimage
from mfunc.lorsal import lorsal
from mfunc.lorsal_test import lorsal_test
from mfunc.BP_message import BP_message
from mfunc.calcError import calcError                           # 导入函数，计算错误率
from mfunc.train_test_random_new import train_test_random_new   # 导入函数，获取训练和测试随机数据的索�?
from mfunc.getNeighFromGrid import getNeighFromGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# load data
dfc_dict = scipy.io.loadmat('libmat/dfc.mat')
img = dfc_dict['dfc']
img = img[1192:1788, :, :]
img = np.delete(img.T, [48, 49], 0).T

# 依次获取img中三个维度的数值，并进行赋值?
sz = img.shape  # shape is (596, 601, 48)
no_lines = sz[0]
no_columns = sz[1]
no_bands = sz[2]
mu = 2
img = img.reshape((sz[0] * sz[1]), sz[2], order='F')    # order = "F" 按照列顺序读取， 默认是行顺序
img = np.transpose(img)

# load groun truth
dfc_gt2_dict = scipy.io.loadmat('libmat/dfc_gt2.mat')
dfc_gt2 = dfc_gt2_dict['dfc_gt2']
groundtruth = dfc_gt2[1192:1788, :]
train_all = np.zeros((2, no_lines * no_columns))
train_all[0, :] = np.arange(1, no_lines * no_columns+1)
train_all[1, :] = np.arange(1, int(no_lines * no_columns)+1)

groundtruth_flatten = groundtruth.flatten('F') # flatten 按列进行展开
train_all[1,:] = groundtruth_flatten           # train_all[1,:] 从第2行所有列被赋值为 groundtruth_flatten中的所有值?

index = []
for i in range(len(train_all[1, :])):
    if train_all[1, i] == 0:
        index.append(i)
        
trainall = np.delete(train_all, index, 1)   # 对所有trainall中第二行中列中等于 0 的进行删除
no_classes = np.amax(trainall[1,:])     # 获取trainall第二行中最大值
print(f' no_classes = {no_classes}') # dobule --> 8.0
# number of training samples per class

numN, nList = getNeighFromGrid(sz[0],sz[1])

Numiter = 1
EMiter = 6
MMiter = 5 # 200
nl = 15

# EM
results_EM_OA_list = []        # 存储Numiter 次EM OA的结果
results_EM_OA_list.clear()
results_EM_MRF_OA_list =[]     # 存储Numiter 次EM MRF OA的结果
results_EM_MRF_OA_list.clear()
for numiter in np.arange(Numiter):
    print('No. Monte Carlo run = %7.0f\n' % (numiter))
    indexes = train_test_random_new(trainall, nl)

    # 根据索引号获取trainall 中的真实数据
    temp = []
    for n in range(indexes.size):
        temp.append( trainall[:, indexes[n]])

    train_set = np.array(temp).T    # 对训练数据转变为xxx类型，并进行转置
    test_set = trainall               # 这种创建会引用方式，不是创建一个新的矩阵，matlab也是引用方式
    test_set = np.delete(test_set, indexes, 1)  # 在所有数据集中，删除包含的训练集数据，剩下的作为测试�?    shape is 2x40

    # 动态调整数值
    dex = train_set[0, :].astype(int).tolist()
    train = img[:, dex]

    # parameters
    sigma = 1.0
    lambda_ = 0.001
    # construct the new feature space for labeled samples
    d, n = train.shape
    train = np.dtype('uint64').type(train)
    tr = train ** 2

    nx = np.sum(tr, axis=0)
    X, Y = np.meshgrid(nx, nx)
    dist = X + Y - (2 * np.transpose(train).dot(train))
    scale = np.mean(dist)
    Ktrain = np.exp(-((((dist) / 2) / scale) / sigma ** 2))

    if True:
        Ktrain_ = np.row_stack((np.ones((n)),Ktrain))  # 在Ktrain（180 * 180） 中增加第一行全为1的值，变成 181 * 180

    # construct the new feature space for the whole image
    img = np.dtype('uint64').type(img)  #  类型转换
    nz = np.sum(img ** 2, axis=0)
    X1, Z1 = np.meshgrid(nz, nx)

    dist1 = Z1 - (2 * np.transpose(train).dot(img)) + X1
    K1all = np.exp(- (dist1 / 2 / scale / sigma ** 2))
    K1all = np.row_stack((np.ones((sz[0] * sz[1])), K1all))

    tag = False
    root_path = r"libmat/val"
    if tag:
        start_time = time.process_time()

    X = Ktrain
    y = train_set[1, :]

    clf = LogisticRegression(penalty='l2', max_iter=2000, solver='saga')      # MMiter = 200
    clf = clf.fit(X, y)

    w_ = np.vstack((clf.intercept_, clf.coef_.T))    # from (181 * 12) shape to (181 * 12) shape

    de_flag = True
    if de_flag:
        for i in np.arange(w_.shape[1]):
            val = w_[:, 0]
            if i >= 1:
                w_[:, i] -= val
        w = np.delete(w_, 0, axis=1)    # 删除第一列
    else:
        for i in np.arange(w_.shape[1]):
            val = w_[:, -1]
            # if i >= 1:
            w_[:, i] -= val
        w = np.delete(w_, -1, axis=1)    # 删除最后列

    y_pred = clf.predict(X)
    print(f'sklearn accuracy score is: {np.around(accuracy_score(y, y_pred), 6)}')

    # # estimate posterior probabilities
    load_p = False   # 导入的数据还是自动生成
    if load_p:
        p = scipy.io.loadmat(os.path.join(root_path, 'p.mat'))['p']
        train_set = scipy.io.loadmat(os.path.join(root_path, 'train_set.mat'))['train_set'] 
    else:
        w, L = lorsal(Ktrain_, train_set[1, :], lambda_, 0.01 * lambda_, MMiter, 0)
        print("L",L)
        p = splitimage(img, train, w, scale, sigma)
    if load_p:
        L_dict =  os.path.join(root_path, 'L.mat')
        L_dicts =  scipy.io.loadmat(L_dict)
        L = L_dicts['L']
        w_dict = scipy.io.loadmat(os.path.join(root_path, 'w.mat'))
        w = w_dict['w']
        scale_d = scipy.io.loadmat(os.path.join(root_path, 'scale.mat'))
        scale = scale_d['scale']
        train = scipy.io.loadmat(os.path.join(root_path, 'train.mat'))['train']
        p = scipy.io.loadmat(os.path.join(root_path, 'p.mat'))['p'] 

    class_results_MLR_map_index = np.argmax(p, axis=0) + 1  # 计算p变量最大值所在的位置，返回index位置， 因为MATLAB从1开始算 所以 + 1 
    class_results_MLR_map = class_results_MLR_map_index.reshape(1, -1) 

    temp_test_set = []
    for i in range(len(test_set[0,:].tolist())):
        temp_test_set.append(int(test_set[0,i]))
    class_results_MLR_map_in = []
    for i in range(len(temp_test_set)):
        class_results_MLR_map_in.append(class_results_MLR_map[numiter, temp_test_set[i]-1])
    class_results_MLR_map_in = np.array(class_results_MLR_map_in).reshape(1, -1)

    ins = np.array(np.arange(1, no_classes+1)).reshape(-1, 1)
    test_set_in = np.array(test_set[1, :] - 1).reshape(1, -1)
    class_results_MLR_OA ,class_results_MLR_kappa ,class_results_MLR_AA ,class_results_MLR_CA, results_MLR_matrix \
        = calcError(test_set_in, class_results_MLR_map_in - 1, ins)

    print(f'class_results_MLR_OA = {np.round(class_results_MLR_OA, 6)}\d')

    if tag:
        print('First part!! ')
        end_time = time.process_time()
        print(f'using {end_time - start_time} seconds!!')

    # calculate interaction potentials
    v0 = np.exp(8)
    v1 = np.round(np.exp(- 2), 4)
    psi = v1 * np.ones((int(no_classes), int(no_classes)))
    for i in np.arange(int(no_classes)):
        psi[i,i] = v0
    psi_temp = np.sum(psi, axis=1)
    # psi_temp = np.sum(psi, axis=1).reshape(1, -1)
    print()
    psi_temp = np.tile(psi_temp, (12, 1))

    psi = psi / psi_temp

    # # calculate beliefs
    p = np.transpose(p)
    belief = BP_message(p, psi, nList, train_set, 5)

    # maxb = np.amax(belief, axis=0)    # 计算p变量中的最大值
    results_DRF_MPM_map = (np.argmax(belief, axis=0) + 1).reshape(1, -1, order='F')  # 计算 belief 变量最大值所在的位置，返回index位置， 因为MATLAB从1开始算 所以 + 1 

    # 生成 calcError 输入数据 results_DRF_MPM_map_in
    results_DRF_MPM_map_in = []
    for i in range(len(temp_test_set)):
        results_DRF_MPM_map_in.append(results_DRF_MPM_map[numiter, temp_test_set[i]-1])
    results_DRF_MPM_map_in = np.array(results_DRF_MPM_map_in).reshape(1, -1)

    class_results_DRF_MPM_OA ,class_results_DRF_MPM_kappa ,class_resultsDRF_MPM_AA ,class_results_DRF_MPM_CA, results_DRF_MPM_matrix \
         = calcError(test_set_in, results_DRF_MPM_map_in - 1, ins)

    # 计算时间
    if tag:
        print('second part!! ')
        end_time = time.process_time()
        print(f'using {end_time - start_time} seconds!!')
        print('-- end second part ----')

    print(f'class_results_DRF_MPM_OA = {np.round(class_results_DRF_MPM_OA, 6)}')


