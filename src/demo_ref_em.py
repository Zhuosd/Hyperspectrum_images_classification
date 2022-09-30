## Demo: Applying DRF-EM to the AVIRIS Indian Pines scene
## RBF with with all bands
## BP for posterior probability estimation

import numpy as np
import scipy

import numpy.matlib
clear('all')
close_('all')
addpath('./GraphCutMex')
addpath(genpath('./SVM'))
# load data
#load TruthMap2;
scipy.io.loadmat('HyperCube_OMIS4Class2')
img = HyperCube_OMIS4Class2
sz = img.shape

no_lines = sz(1)
no_columns = sz(2)
no_bands = sz(3)
mu = 2
# load groun truth
scipy.io.loadmat('TruthMap2')
groundtruth = TruthMap2
train_all = np.zeros((2,no_lines * no_columns))
train_all[1,:] = np.arange(1,no_lines * no_columns+1)
train_all[2,:] = TruthMap2
train_all[:,train_all[2,:] == 0] = []
trainall = train_all
no_classes = np.amax(trainall(2,:))

#load  the neighborhood from the grid, in this paper we use the first order neighborhood
scipy.io.loadmat('belief_propagation_Zaoyuan_NeighFormGrid_full','numN','nList')
#save  belief_propagation_Zaoyuan_NeighFormGrid_full numN nList
# # compute the neighborhood from the grid, in this paper we use the first order neighborhood
#[numN, nList] = getNeighFromGrid(sz(1), sz(2));

## parameter settings
Numiter = 5

EMiter = 6

MMiter = 200

nl = 15

no_samples = no_lines * no_columns
img = ToVector(img)
img = np.transpose(img)
## EM
for numiter in np.arange(1,Numiter+1).reshape(-1):
    print('No. Monte Carlo run = %7.0f\n' % (numiter))
    ## genetate the labeled training set and the test set
# generate labeled training set by randomly selecting samples from the ground truth
    indexes = train_test_random_new(trainall(2,:),nl)
    train_set = trainall(:,indexes)
    ntrain = train_set.shape[2-1]
    # the rest of the ground truth are for testing
    test_set = trainall
    test_set[:,indexes] = []
    # # save the labeled training set and the test set
# save(['Train_test_set/Zaoyuan_5perClass_',num2str(numiter),'.mat'],'train_set','test_set');
    ## construct new feature spaces using the RBF kernel and labeled samples
    train = img(:,train_set(1,:))
    # parameters
    sigma = 1.0
    lambda_ = 0.001
    # construct the new feature space for labeled samples
    d,n = train.shape
    nx = sum(train ** 2)
    X,Y = np.meshgrid(nx)
    dist = X + Y - 2 * np.transpose(train) * train
    scale = mean(dist)
    Ktrain = np.exp(- dist / 2 / scale / sigma ** 2)
    Ktrain = np.array([[np.ones((1,n))],[Ktrain]])
    # construct the new feature space for the whole image
    nz = sum(img ** 2)
    X1,Z1 = np.meshgrid(nz,nx)
    dist1 = Z1 - 2 * np.transpose(train) * img + X1
    K1all = np.exp(- dist1 / 2 / scale / sigma ** 2)
    K1all = np.array([[np.ones((1,137 * 202))],[K1all]])
    clear('X1','Z1','dist1','dist','X','Y','nx','nz')
    ## generate MLR results
# estimate w using the LORSAL algorithm and labeled training set in a
# supervised learning manner
    w,L = LORSAL(Ktrain,train_set(2,:),lambda_,0.01 * lambda_,MMiter)
    # estimate posterior probabilities
    p = splitimage2(img,train,w,scale,sigma)
    clear('train')
    # results
    __,class_results_MLR.map[numiter,:] = np.amax(p)
    class_results_MLR.OA[numiter],class_results_MLR.kappa[numiter],class_results_MLR.AA[numiter],class_results_MLR.CA[numiter,:] = calcError(test_set(2,:) - 1,class_results_MLR.map(numiter,test_set(1,:)) - 1,np.array([np.arange(1,no_classes+1)]))
    print('class_results_MLR.OA = %7.3f\n' % (class_results_MLR.OA(numiter)))
    ## generate MLR-MRF results
    Dc = np.reshape(np.transpose(np.log(p + eps)), tuple(np.array([sz(1),sz(2),no_classes])), order="F")
    Sc = np.ones((no_classes,no_classes)) - np.eye(no_classes)
    gch = GraphCut('open',- Dc,2 * Sc)
    gch,segmap = GraphCut('expand',gch)
    seg_results_MLL.map[numiter,:] = segmap
    gch = GraphCut('close',gch)
    seg_results_MLL.OA[numiter],seg_results_MLL.kappa[numiter],seg_results_MLL.AA[numiter],seg_results_MLL.CA[numiter,:] = calcError(test_set(2,:) - 1,seg_results_MLL.map(numiter,test_set(1,:)),np.array([np.arange(1,no_classes+1)]))
    print('seg_results_MLL.OA = %7.3f\n' % (seg_results_MLL.OA(numiter)))
    ## generate LBP-MPM resutls
# calculate interaction potentials
    v0 = np.exp(2)
    v1 = np.exp(0)
    psi = v1 * np.ones((no_classes,no_classes))
    for i in np.arange(1,no_classes+1).reshape(-1):
        psi[i,i] = v0
    psi_temp = sum(psi)
    psi_temp = np.matlib.repmat(psi_temp,no_classes,1)
    psi = psi / psi_temp
    clear('psi_temp')
    # calculate beliefs
    p = np.transpose(p)
    belief = BP_message(p,psi,nList,train_set,5)
    # results
    maxb,class_results_MPM.map[numiter,:] = np.amax(belief)
    class_results_MPM.OA[numiter],class_results_MPM.kappa[numiter],class_results_MPM.AA[numiter],class_results_MPM.CA[numiter,:] = calcError(test_set(2,:) - 1,class_results_MPM.map(numiter,test_set(1,:)) - 1,np.array([np.arange(1,no_classes+1)]))
    print('class_results_MPM.OA  = %7.3f\n' % (class_results_MPM.OA(numiter)))
    ## generate EM-MRF resutls
# EM iterations start
    for emiter in np.arange(1,EMiter+1).reshape(-1):
        ## updating w using the whole image and beliefs (soft labels) in a semisupervised manner
        w,p = LORSAL_BP_semi20171023(K1all,train_set(2,:),belief,eps,lambda_,0.001 * lambda_,MMiter,w)
        # generate new MLR results using the new learned w
        __,class_results_DRF_MLR.map[numiter,emiter,:] = np.amax(p)
        class_results_DRF_MLR.OA[numiter,emiter],class_results_DRF_MLR.kappa[numiter,emiter],class_results_DRF_MLR.AA[numiter,emiter],class_results_DRF_MLR.CA[numiter,emiter,:] = calcError(test_set(2,:) - 1,np.transpose(np.squeeze(class_results_DRF_MLR.map(numiter,emiter,test_set(1,:)))) - 1,np.array([np.arange(1,no_classes+1)]))
        ## generate new LBP results using the new learned w
# calculate beliefs
        p = np.transpose(p)
        belief = BP_message(p,psi,nList,train_set,5)
        # results
        __,class_results_DRF_MPM.map[numiter,emiter,:] = np.amax(belief)
        class_results_DRF_MPM.OA[numiter,emiter],class_results_DRF_MPM.kappa[numiter,emiter],class_results_DRF_MPM.AA[numiter,emiter],class_results_DRF_MPM.CA[numiter,emiter,:] = calcError(test_set(2,:) - 1,np.transpose(np.squeeze(class_results_DRF_MPM.map(numiter,emiter,test_set(1,:)))) - 1,np.array([np.arange(1,no_classes+1)]))
        # fprintf('class_results_DRF_MPM.OA  = #7.3f\n', squeeze(class_results_DRF_MPM.OA))
    # print EM-DRF results
    type_str = np.matlib.repmat('%7.3f ',1,emiter)
    np.array(['class_results_DRF_MLR.OA = ',type_str,'\n']).write(class_results_DRF_MLR.OA(numiter,:) % ())
    np.array(['class_results_DRF_MPM.OA = ',type_str,'\n']).write(class_results_DRF_MPM.OA(numiter,:) % ())

mean(class_results_MLR.OA)
mean(class_results_MPM.OA)
mean(class_results_DRF_MLR.OA)
mean(class_results_DRF_MPM.OA)
save('Results_Zaoyuan_MLR_All','class_results_MLR','class_results_MPM','class_results_DRF_MLR','class_results_DRF_MPM')