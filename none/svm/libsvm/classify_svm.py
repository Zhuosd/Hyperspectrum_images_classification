import numpy as np
    
def classify_svm(varargin = None): 
    #CLASSIFYSVM Classify with libSVM an image
    
    #		[outdata, out_param] = classify_svm(img, train, opt)
    
    # INPUT
#   img    Multispectral image to be classified.
#   train  Training set image (zero is unclassified and will not be
#           considered).
#   opt    input parameters. Structure with each field correspondent to a
#           libsvm parameter
#           Below the availabel fields. The letters in the brackets corresponds to the flags used in libsvm:
#             "svm_type":	(-s) set type of SVM (default 0)
#                   0 -- C-SVC
#                   1 -- nu-SVC
#     	            2 -- one-class SVM
#     	            3 -- epsilon-SVR
#     	            4 -- nu-SVR
#             "kernel_type": (-t) set type of kernel function (default 2)
#                   0 -- linear: u'*v
#                   1 -- polynomial: (gamma*u'*v + coef0)^degree
#                   2 -- radial basis function: exp(-gamma*|u-v|^2)
#                   3 -- sigmoid: tanh(gamma*u'*v + coef0)
#                   4 -- precomputed kernel (kernel values in training_instance_matrix)
#             "kernel_degree": (-d) set degree in kernel function (default 3)
#             "gamma": set gamma in kernel function (default 1/k, k=number of features)
#             "coef0": (-r) set coef0 in kernel function (default 0)
#             "cost": (-c) set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
#             "nu": (-n) parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
#             "epsilon_regr": (-p) set the epsilon in loss function of epsilon-SVR (default 0.1)
#             "chache": (-m) set cache memory size in MB (default 100)
#             "epsilon": (-e) set tolerance of termination criterion (default 0.001)
#             "shrinking": (-h) whether to use the shrinking heuristics, 0 or 1 (default 1)
#             "probability_estimates": (-b) whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
#             "weight": (-wi) set the parameter C of class i to weight*C, for C-SVC (default 1)
#             "nfold": (-v) n-fold cross validation mode
#             "quite": (-q) quiet mode (no outputs)
#           For setting other default values, modify generateLibSVMcmd.
    
    # OUTPUT
#   outdata    Classified image
#   out_param  structure reports the values of the parameters
    
    # DESCRIPTION
# This routine classify an image according to the training set provided
# with libsvm. By default, the data are scaled and normalized to have unit
# variance and zero mean for each band of the image. If the parameters
# defining the model of the svm (e.g., the cost C and gamma) are not
# provided, the function call the routin MODSEL and which optimizes the
# parameters. Once the model is trained the image is classified and is
# returned as output.
    
    # SEE ALSO
# EPSSVM, MODSEL, GETDEFAULTPARAM_LIBSVM, GENERATELIBSVMCMD, GETPATTERNS
    
    # Mauro Dalla Mura
# Remote Sensing Laboratory
# Dept. of Information Engineering and Computer Science
# University of Trento
# E-mail: dallamura@disi.unitn.it
# Web page: http://www.disi.unitn.it/rslab
    
    # Parse inputs
    if len(varargin) == 2:
        data_set = varargin[0]
        train = varargin[2]
        in_param = struct
        #    in_param.kernel_type = 2;   # default RBF
    else:
        if len(varargin) == 3:
            data_set = varargin[0]
            train = varargin[2]
            in_param = varargin[3]
    
    # Default Parameters - Scaling the data
    scaling_range = True
    
    scaling_std = True
    
    # Read in_param
    if (isfield(in_param,'scaling_range')):
        scaling_range = in_param.scaling_range
    else:
        in_param.scaling_range = scaling_range
    
    if (isfield(in_param,'scaling_std')):
        scaling_std = in_param.scaling_std
    else:
        in_param.scaling_std = scaling_std
    
    # ------------------------
    
    nrows,ncols,nfeats = data_set.shape
    Ximg = double(reshape(data_set,nrows * ncols,nfeats))
    # Transform training set in a format compliant to RF
    X,L = getPatterns(data_set,train)
    nclasses = len(unique(L))
    #[X,row_factor] = removeconstantrows(X);   # Remove redundant features
# Ximg = Ximg(:,row_factor.keep); # Remove redundant features
    
    Ximg = X
    # ========= Preprocessing =========
# Scale each feature of the data in the range [-1,1]
    if (scaling_range):
        X,scale_factor = mapminmax(X)
        nfold = 10
        nelem = np.round(Ximg.shape[1-1] / nfold)
        for i in np.arange(1,nfold - 1+1).reshape(-1):
            Ximg[np.arange[[i - 1] * nelem + 1,i * nelem+1],:] = np.transpose((mapminmax('apply',np.transpose(Ximg(np.arange((i - 1) * nelem + 1,i * nelem+1),:)),scale_factor)))
        Ximg[np.arange[[nfold - 1] * nelem + 1,end()+1],:] = np.transpose((mapminmax('apply',np.transpose(Ximg(np.arange((nfold - 1) * nelem + 1,end()+1),:)),scale_factor)))
    
    # Scale each feature in order to have std=1
    if (scaling_std):
        X,scale_factor = mapstd(X)
        nfold = 5
        nelem = np.round(Ximg.shape[1-1] / nfold)
        for i in np.arange(1,nfold - 1+1).reshape(-1):
            Ximg[np.arange[[i - 1] * nelem + 1,i * nelem+1],:] = np.transpose((mapstd('apply',np.transpose(Ximg(np.arange((i - 1) * nelem + 1,i * nelem+1),:)),scale_factor)))
        Ximg[np.arange[[nfold - 1] * nelem + 1,end()+1],:] = np.transpose((mapstd('apply',np.transpose(Ximg(np.arange((nfold - 1) * nelem + 1,end()+1),:)),scale_factor)))
    
    tic
    # Train the model
    model,out_param = epsSVM(np.transpose(double(X)),np.transpose(double(L)),in_param)
    out_param.time_tr = toc
    out_param.nfeats = len(row_factor.keep)
    # Classify the whole data
#Ximg = double(reshape(data_set, nrows*ncols, nfeats));
    
    
    # nfold = 5;
# nelem = round(size(Ximg,1)/nfold);
    
    # for i=1:nfold-1
#     Ximg((i-1)*nelem+1:i*nelem,:) = (mapminmax('apply',Ximg((i-1)*nelem+1:i*nelem,:)',scale_factor))';
# end
# Ximg((nfold-1)*nelem+1:end,:) = (mapminmax('apply',Ximg((nfold-1)*nelem+1:end,:)',scale_factor))';
    
    #Ximg = Ximg*scale_factor;
    cmd = generateLibSVMcmd(out_param,'predict')
    
    if len(cmd)==0:
        predicted_labels,out_param.accuracy = svmpredict(np.ones((nrows * ncols,1)),Ximg,model)
    else:
        predicted_labels,out_param.accuracy,out_param.prob_estimates = svmpredict(np.ones((nrows * ncols,1)),Ximg,model,cmd)
    
    # reshape the array of labels to the original dimensions of the image
    outdata = reshape(predicted_labels,nrows,ncols,1)
    out_param.time_tot = toc
    return outdata,out_param