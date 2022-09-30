## Demo: Applying SVM-MRF to the AVIRIS Indian Pines scene
## Pines scene by using LORSAL only and with MRF for post-processing
## purposes, respectively.
##


import scipy
import numpy as np
import scipy.io as io
from libsvm.svmutil import *
from mfunc.calcError import calcError                           # 导入函数，计算错误率
from mfunc.train_test_random_new import train_test_random_new   # 导入函数，获取训练和测试随机数据的索引   

def SVM_MRF_Demo():

    # load data
    #load ZaoyuanMap2;
    # 导入libmat文件夹中的mat文件
    ho_mat_dicts = scipy.io.loadmat('libmat/HyperCube_OMIS4Class2.mat')
    img = ho_mat_dicts['HyperCube_OMIS4Class2']     # 加载key为 HyperCube_OMIS4Class2 的数值
    # 依次获取img中三个维度的数值，并进行赋值
    sz = img.shape
    no_lines = sz[0]
    no_columns = sz[1]
    no_bands = sz[2]
    mu = 2

    # load groun truth
    # scipy.io.loadmat('TruthMap2')
    tm_mat_dicts = scipy.io.loadmat('libmat/TruthMap2.mat')
    groundtruth = tm_mat_dicts['TruthMap2']
    train_all = np.zeros((2, no_lines * no_columns))    # 创建2 * (no_lines * no_columns)矩阵
    # print(f'train_all -- {train_all}')
    # print(f'train_all.shape -- {train_all.shape}')
    # print('#'*30)
    # print(int(no_lines * no_columns))  #  27674
    train_all[0, :] = np.arange(1, no_lines * no_columns+1) # train_all 中第一行所有列，赋值为 从(1 到no_lines * no_columns+1)有序数
    train_all[1, :] = np.arange(1, int(no_lines * no_columns)+1)
    # print(f'train_all -- {train_all}')
    # print(f'train_all.shape -- {train_all.shape}')
    # print('#'*30)

    groundtruth_flatten = groundtruth.flatten('F') # flatten 按列进行展开
    # print((groundtruth_flatten[26851:26853]))
    # print(groundtruth_flatten.shape)
    train_all[1,:] = groundtruth_flatten           # train_all[1,:] 第2行所有列被赋值为 groundtruth_flatten中的所有值
    # print(f'train_all -- {train_all}')
    # print(f'train_all.shape -- {train_all.shape}')

    ########################################################
    # 获取train_all 第二行所有列值依次等于(1 到len(train_all[1, :])) 值为0 的数的列号
    index = []
    for i in range(len(train_all[1, :])):
        if train_all[1, i] == 0:
            index.append(i)
            
    trainall = np.delete(train_all, index, 1)   # 对所有trainall中第二行中列为0 的进行删除
    # print(trainall)
    # print(trainall.shape)
    # print(' trainall '*10)
    # print(len(train_all[1, :]) - len(index))
    #########################################################

    no_classes = np.amax(trainall[1,:])     # 获取trainall第二行中最大值
    # print(no_classes) # dobule --> 8.0
    # number of training samples per class
    no_class = 5
    Numiter = 1
    no_samples = no_lines * no_columns

    '''
    function v = ToVector(im)
    % takes MxNx3 picture and returns (MN)x3 vector
    sz = size(im);
    v = reshape(im, [prod(sz(1:2)) sz(3)]);
    '''

    ########################################################
    # 把 img三维的变成两维，按照先列读取方式
    sz = img.shape
    img = img.reshape((sz[0] * sz[1]), sz[2], order='F')
    ########################################################
    # print(img)
    # print(img.shape)
    # print('-'*30)

    img = np.transpose(img)     # 转置
    # print(img)
    # print(img.shape)
    train_all = trainall
    # print(train_all)
    # print(train_all.shape)
    # print('-train_all-'*10)

    # from libsvm.svmutil import *
    # from mfunc.train_test_random_new import train_test_random_new
    # from sklearn.model_selection import train_test_split

    n = int(Numiter)
    # K = int(np.amax(trainall[1:, :]))   # K is 8
    # indexes = np.matrix([])
    # for numiter in np.arange(1):
    results_SVM_OA_list = []        # 存储Numiter 次SVM OA的结果
    results_SVM_OA_list.clear()
    results_SVM_MRF_OA_list =[]     # 存储Numiter 次SVM MRF OA的结果
    results_SVM_MRF_OA_list.clear()
    for numiter in np.arange(Numiter):
        print('No. Monte Carlo run = %7.0f ' % (numiter))
        # randomly disctribute the ground truth image to training set and test set
        # randomly generate the training and test sets
        # print('start'*10)
        # 获取trainall 中的随机索引号
        indexes = train_test_random_new(trainall, 5)
        # print('#'*30)
        # print(indexes)
        # print(indexes.shape)  
        # print('over'*10)  

        #  is error method
        # len_in = indexes.size
        # # print(trainall.shape)
        # train_set = trainall[:, :len_in]
        # print(train_set)
        # print(train_set.shape)
        # print('9'*30)
        # 根据索引号获取trainall 中的真实数据
        temp = []
        for n in range(indexes.size):
            temp.append( trainall[:, indexes[n]])
    
        train_set = np.matrix(temp).T    # 对训练数据转变为matrix类型，并进行转置
        # print((train_set))
        # print((train_set.shape))
        # print('-'*35)

    #     test_set = trainall               # 这种创建会引用方式，不是创建一个新的矩阵，matlab也是引用方式
        test_set = np.matrix(trainall)      # 新创建一个矩阵 
    #     test_set[:,indexes] = []
        test_set = np.delete(test_set, indexes, 1)  # 在所有数据集中，删除包含的训练集数据，剩下的作为测试集    shape is 2x40
        # print((test_set))
        # print((test_set.shape))
        # print('-'*35)
        train_samples = np.matrix(train_set)    # train_samples shape is  2x40
        # print((train_samples))
        # print((train_samples.shape))
        # print('-'*35)
        # print(id(train_set))
        train_vectors = img[:, :train_set[0,:].size]    #   shape is 80x40， 取出img中前40列的所有行
        # print(train_vectors)
        # print(train_vectors.shape)
        # print('#' * 30)

        # XX = np.matrix(np.zeros([sz[0] * sz[1], sz[2]]))
        # h = int(sz[0] * sz[1])
        # w = int(sz[2])
        # print(w, h)
        # XX = np.matrix(np.zeros((w, h)))
        #  进行图像数据的归一化处理
        temp = []
        for k in np.arange(no_bands):
            temp.append(np.double(img[k,:] - np.amin(img[k,:])) / (np.amax(img[k, : ]) - np.amin(img[k, : ])))
            # XX = np.insert(XX, k,  ((img[k,:] - np.amin(img[k,:])) / (np.amax(img[k, : ]) - np.amin(img[k, : ]))), axis=0) 

        # print(XX.reshape(w, h))
        XX = np.array((temp))
        # print((temp))

        # print(XX)
        # print(XX.shape)
        # print('#' * 30)
        # print(img)
        # print(img.shape)

        # trainsamples=XX(:,train_samples(1,:));
        # model = svmtrain(np.transpose(train_samples[2, :]),np.transpose(XX(:,train_samples(1,:))),'-s 0 -c 125 -g 2^(-6) -b 1')
        # 适应python中类型，
        y_list = (train_samples[1, :]).tolist() # train_samples = train_set ,the shape is(1, 40)
        # y = np.transpose(train_samples[1, :])   # shape = (40, 1)
        # print(len(y_list[0]))
        # print(' y ' * 10)

        # x = np.transpose(XX[:,train_samples[0, :]])
        x_list = (train_samples[0, :].tolist())[0]
        x_list_int = list(map(int, x_list))
        x = np.transpose(XX[:, x_list_int])   # shape = (80, 40)
        # print(x)
        # print(x.shape)
        # print('#' * 30)
        x = x.tolist()
        # print(x)
        # print(len(x))
        # g = 2**(-6)  #  0.0156
        param = svm_parameter(f'-s 0 -c 125 -g {2**(-6)} -b 1')
        prob = svm_problem(y_list[0], x)
        #  模型训练
        model = svm_train(prob, param)
        # model = svm_train(y_list[0], x, param)
        print(' - ' * 30)
        # print(type(model))
        # print(model.nr_class)
    


        # # groupX = randi(no_classes,no_samples,1)
        groupX = np.random.randint(1, no_classes, no_samples)   # 随机生成1 到no_classes范围的 no_samples个数值
        # print(groupX)
        y_test = groupX     # 作为svm预测的y 标签参数
        x_test = np.transpose(XX).tolist()      # 转置并转为list python相应类型 ， x_test len is 27674
        # print(x_test)
        # print(len(y_test))  # len is 27674

        
        # predict_label,accuracy,prob_estimates = svmpredict(groupX,np.transpose(XX),model,'-b 1')
        predict_label, accuracy, prob_estimates = svm_predict(y_test, x_test, model,'-b 1')
        # print(predict_label, accuracy, prob_estimates)
        
        # print((prob_estimates[:3])) # 打印前三个，prob_estimates 27674 个list， 每个list里面有8个元素
        prob_estimates = np.matrix(prob_estimates)      # prob_estimates shape is (27674, 8)
        # print((prob_estimates).shape)

        # 保存模型中包含了rho  svm_type c_svc  kernel_type rbf gamma 0.015625  nr_class 8 total_sv 35  等信息
        # svm_save_model('results_save//model',model)
        # print('finished saving model!')
        p = np.transpose(prob_estimates)
        # print(prob_estimates.shape)
        
        cmap = p.argmax(0)          # 每一列中的最大值, 返回的是下标值， python 从0开始
        # print((cmap))             #  [[4 4 4 ... 6 6 6]]
        # print((cmap).shape)
        cmap = (cmap.tolist())[0]   # 展成一维，并转化为list
        # print(len(cmap))

        # from mfunc.calcError import calcError

        temp_test_set = []
        for i in range(len(test_set[0,:].tolist()[0])):
            temp_test_set.append(int(test_set[0,i]))
        # # print(temp_test_set[-10:])
        # print(len(temp_test_set))
        
        SVM = []
        for i in range(len(temp_test_set)):
            SVM.append(cmap[temp_test_set[i]-1])

        # print(SVM[-26:])
        # print(type(SVM))
        # print(type(SVM[0]))
        # print(len(SVM))


        # tes = test_set[0,:].flatten('F')[0]
        # print(len(test_set[0,:].tolist()[0]))
        # print((test_set[0,:].tolist())[0][-10:])
        # print(len(tes[0]))
        # print((tes))

        # print(np.array(test_set[1,:]) - 1, (np.array([test_set[0,:]])) - 1)
        # print(np.array(test_set[1,:]) - 1, (np.array(cmap[test_set[0,:]])) - 1)
        
        results_svm_OA, results_svm_kappa ,results_svm_AA , results_svm_CA, results_svm_matrix = calcError(np.array(test_set[1,:] - 1), np.array(SVM), np.array(np.arange(1, no_classes+1)))
        results_SVM_OA_list.append(results_svm_OA)
        print('class_results_svm_OA = %7.3f\n' % (results_svm_OA))

        # ## post-processing with MRF     #  因为 GraphCut 函数没有找到对应的python调用接口，故此部分进行注释处理（已转换） 
        # P = np.transpose(np.log(p + np.spacing(1)))     ## (27674, 8)
        # # print(P)
        # # print(P.shape)  
        # P = np.array(P)     #  shape too large to be a matrix.  (27674, 8)
        # Dc = P.reshape(sz[0], sz[1], int(no_classes), order="F")    # (137, 202, 8)
        # # print(Dc.shape)
        # Sc = np.ones((int(no_classes),int(no_classes)))  - np.eye(int(no_classes))  # （8 * 8）单位方阵，令对角线为0
        ################ 因为 GraphCut 函数没有找到对应的python调用接口，故此部分暂未能输出对应的结果   #################################
        # gch = GraphCut('open',- Dc,mu * Sc)
        # gch,map_MRF = GraphCut('expand',gch)
        # gch = GraphCut('close',gch)
        # gch_x, gch_y = gch.shape 
        # gch = gch.reshape((gch_x * gch_y), 1)
        # SVM_MRF = []
        # for i in range(len(temp_test_set)):
        #     SVM_MRF.append(gch[temp_test_set[i]-1])
        # results_svm_mrf_OA, results_svm_mrf_kappa, results_svm_mrf_AA, results_svm_mrf_CA, results_svm_mrf_matrix = calcError(np.array(test_set[1,:] - 1), np.array(SVM_MRF), np.array(np.arange(1, no_classes+1)))
        # results_SVM_MRF_OA_list.append(results_svm_mrf_OA)
        # print('class_results_SVM_MRF.OA = %7.3f\n' % (results_svm_mrf_OA))

    # mean_SVM_OA = np.mean(results_SVM_OA_list)    # 取存储的n次结果中的均值
    # mean_SVM_MRF_OA = np.mean(results_SVM_MRF_OA_list)
    print(' - ' * 30)
    print(f'The results of {n} class_results_svm_OA is : \n {results_SVM_OA_list}')
    # print(f'The results of {n} class_results_MRF_svm_OA is \n {results_SVM_MRF_OA_list}')
    # scipy.io.savemat('results_save//results_svm.mat', {'results_svm': results_SVM_OA_list})
    # scipy.io.savemat('results_save//results_svm_mrf.mat', {'results_svm_mrf': results_SVM_MRF_OA_list})


if __name__ == '__main__':
    SVM_MRF_Demo()





