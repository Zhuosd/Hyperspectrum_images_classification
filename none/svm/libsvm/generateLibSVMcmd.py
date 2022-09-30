import numpy as np
    
def generateLibSVMcmd(options = None,call = None): 
    #GENERATELIBSVMCMD Generate the command line as a string that will be passed for calling libsvm
    
    #		[cmd] = generateLibSVMcmd(options, call)
    
    # INPUT
#   options input parameters. Structure with each field correspondent to a
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
#   call  string identifying the purpose of the generation of the cmd.
#       Availble options are the following.
#       'modsel':   for performing the model selection of the svm ('nfold'
#           option is enabled). When nfold is enabled, svmtrain return the
#           value of crossvalidation accuracy.
#       'train':    for training the svm ('nfold' option is disabled). With
#           this setting, libsvm returns a trained model.
#       'predict':  for predicting the labels of data.
    
    # OUTPUT
#   cmd    string passed to libsvm if 'call' is defined, otherwise the
#           structure with default values for the parameters is returned.
    
    # DESCRIPTION
# This routine takes as input a structure whose fields represent the values
# of some parameters of libsvm and gives as output a string containing the
# options compliant for being used by libsvmtrain and libsvmpredict. If the
# input variable 'call' is 'train', then the string for training is given,
# if it is 'predict' it will be the string for predict (it is needed if the
# option relative to the generation of the probabilities is defined). If
# 'call' is not defined then the structure with default values is returned.
# The default values can be simply changed by uncommeting the line of the
# desired parameter at the beginning of the file and setting the new
# default value.
    
    # SEE ALSO
# EPSSVM, MODSEL, GETDEFAULTPARAM_LIBSVM, CLASSIFY_SVM, GETPATTERNS
    
    # $Id$
    
    # Mauro Dalla Mura
# Remote Sensing Laboratory
# Dept. of Information Engineering and Computer Science
# University of Trento
# E-mail: dallamura@disi.unitn.it
# Web page: http://www.disi.unitn.it/rslab
    
    # # Set default values. Uncomment the line of the variable of interest and
# # set the desired default value.
# s = 0;      # C-SVC
# t = 2;      # Kernel RBF
# d = 3;      # degree in kernel function
# # g = 1/k;    # set gamma in kernel function
# r = 0;      # coef0 in kernel function
# c = 1;      # parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
# n = 0.5;    # parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
# p = 0.1;    # epsilon in loss function of epsilon-SVR (default 0.1)
# m = 100;    # cache memory size in MB (default 100)
# e = 0.001;  # tolerance of termination criterion (default 0.001)
# h = 1;      # whether to use the shrinking heuristics, 0 or 1 (default 1)
    b = 1
    
    # wi = 1;     # weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
# v = 5;      # n-fold cross validation mode
# q = 1;      # quiet mode (no outputs)
    
    if len(varargin) == 0:
        # Check if a variable is already defined (if it is already defined means that it was
# uncommented at the beginning of this file and so its default value has changed).
        if (not ('svm_type' is not None) ):
            svm_type = 0
        if (not ('kernel_type' is not None) ):
            kernel_type = 2
        if (not ('kernel_degree' is not None) ):
            kernel_degree = 3
        #         if (~exist('gamma','var'))
#             gamma = 1/k; # with k number of features
#         end
        if (not ('coef0' is not None) ):
            coef0 = 0
        if (not ('cost' is not None) ):
            cost = 1
        if (not ('nu' is not None) ):
            nu = 0.5
        if (not ('epsilon_regr' is not None) ):
            epsilon_regr = 0.1
        if (not ('chache' is not None) ):
            chache = 100
        if (not ('epsilon' is not None) ):
            epsilon = 0.001
        if (not ('shrinking' is not None) ):
            shrinking = 1
        if (not ('probability_estimates' is not None) ):
            probability_estimates = 0
        if (not ('nfold' is not None) ):
            nfold = 5
        if (not ('weight' is not None) ):
            weight = 1
        if (not ('quiet' is not None) ):
            quiet = 1
        options = struct
        # Check if a variable is present in the structure, if so it means
# that the user has define it
        if (not isfield(options,'svm_type') ):
            options.svm_type = svm_type
        if (not isfield(options,'kernel_type') ):
            options.kernel_type = kernel_type
        if (not isfield(options,'kernel_degree') ):
            options.kernel_degree = kernel_degree
        #         if (~isfield(options, 'gamma'))
#             options.gamma = gamma;
#         end
        if (not isfield(options,'coef0') ):
            options.coef0 = coef0
        if (not isfield(options,'cost') ):
            options.cost = cost
        if (not isfield(options,'nu') ):
            options.nu = nu
        if (isfield(options,'epsilon_regr')):
            options.epsilon_regr = epsilon_regr
        if (not isfield(options,'chache') ):
            options.chache = chache
        if (not isfield(options,'epsilon') ):
            options.epsilon = epsilon
        if (isfield(options,'shrinking')):
            options.shrinking = shrinking
        if (not isfield(options,'probability_estimates') ):
            options.probability_estimates = probability_estimates
        if (not isfield(options,'nfold') ):
            options.nfold = nfold
        if (not isfield(options,'weight') ):
            options.weight = weight
        if (not isfield(options,'quiet') ):
            options.quiet = quiet
        cmd = options
    else:
        if (len(varargin) == 1) and (isstruct(options)):
            call = 'none'
    
    if (np.logical_or(str(call) == str('train'),str(call) == str('modsel'))) == 1:
        #   "Usage: model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');\n"
# 	"libsvm_options:\n"
# 	"-s svm_type : set type of SVM (default 0)\n"
# 	"	0 -- C-SVC\n"
# 	"	1 -- nu-SVC\n"
# 	"	2 -- one-class SVM\n"
# 	"	3 -- epsilon-SVR\n"
# 	"	4 -- nu-SVR\n"
# 	"-t kernel_type : set type of kernel function (default 2)\n"
# 	"	0 -- linear: u'*v\n"
# 	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
# 	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
# 	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
# 	"	4 -- precomputed kernel (kernel values in training_instance_matrix)\n"
# 	"-d degree : set degree in kernel function (default 3)\n"
# 	"-g gamma : set gamma in kernel function (default 1/k)\n"
# 	"-r coef0 : set coef0 in kernel function (default 0)\n"
# 	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
# 	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
# 	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
# 	"-m cachesize : set cache memory size in MB (default 100)\n"
# 	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
# 	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
# 	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
# 	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
# 	"-v n : n-fold cross validation mode\n"
# 	"-q : quiet mode (no outputs)\n"
        # Read values if are passed in the structure. If a variable was already
# present (e.g., default value) then it will be overwritten.
        if (isfield(options,'svm_type')):
            svm_type = options.svm_type
        if (isfield(options,'kernel_type')):
            kernel_type = options.kernel_type
        if (isfield(options,'kernel_degree')):
            kernel_degree = options.kernel_degree
        if (isfield(options,'gamma')):
            gamma = options.gamma
        if (isfield(options,'coef0')):
            coef0 = options.coef0
        if (isfield(options,'cost')):
            cost = options.cost
        if (isfield(options,'nu')):
            nu = options.nu
        if (isfield(options,'epsilon_regr')):
            epsilon_regr = options.epsilon_regr
        if (isfield(options,'chache')):
            chache = options.chache
        if (isfield(options,'epsilon')):
            epsilon = options.epsilon
        if (isfield(options,'shrinking')):
            shrinking = options.shrinking
        if (isfield(options,'probability_estimates')):
            probability_estimates = options.probability_estimates
        if (isfield(options,'nfold')):
            nfold = options.nfold
        if (isfield(options,'weight')):
            weight = options.weight
        if (isfield(options,'quiet')):
            quiet = options.quiet
        cmd = ''
        # Default values
        if (('svm_type' is not None)):
            cmd = np.array([cmd,'-s ',num2str(svm_type),' '])
        if (('kernel_type' is not None)):
            cmd = np.array([cmd,'-t ',num2str(kernel_type),' '])
        if (('kernel_degree' is not None)):
            cmd = np.array([cmd,'-d ',num2str(kernel_degree),' '])
        if (('gamma' is not None)):
            cmd = np.array([cmd,'-g ',num2str(gamma),' '])
        if (('coef0' is not None)):
            cmd = np.array([cmd,'-r ',num2str(coef0),' '])
        if (('cost' is not None)):
            cmd = np.array([cmd,'-c ',num2str(cost),' '])
        if (('nu' is not None)):
            cmd = np.array([cmd,'-n ',num2str(nu),' '])
        if (('epsilon_regr' is not None)):
            cmd = np.array([cmd,'-p ',num2str(epsilon_regr),' '])
        if (('chache' is not None)):
            cmd = np.array([cmd,'-m ',num2str(chache),' '])
        if (('epsilon' is not None)):
            cmd = np.array([cmd,'-e ',num2str(epsilon),' '])
        if (('shrinking' is not None)):
            cmd = np.array([cmd,'-h ',num2str(shrinking),' '])
        if (('probability_estimates' is not None)):
            cmd = np.array([cmd,'-b ',num2str(probability_estimates),' '])
        if (('nfold' is not None) and str(call) == str('modsel')):
            cmd = np.array([cmd,'-v ',num2str(nfold),' '])
        if (('weight' is not None)):
            cmd = np.array([cmd,'-wi ',num2str(weight),' '])
        if (('quiet' is not None)):
            cmd = np.array([cmd,'-q ',num2str(quiet),' '])
    else:
        if str(call) == str('predict') == 1:
            #           libsvm_options:\n"
# 		"    -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
            cmd = ''
            if (isfield(options,'probability_estimates')):
                cmd = np.array(['-b ',num2str(options.probability_estimates)])
        else:
            # Check if a variable is already defined (if it is already defined means that it was
# uncommented at the beginning of this file and so its default value has changed).
            if (not ('svm_type' is not None) ):
                svm_type = 0
            if (not ('kernel_type' is not None) ):
                kernel_type = 2
            if (not ('kernel_degree' is not None) ):
                kernel_degree = 3
            #         if (~exist('gamma','var'))
#             gamma = 1/k; # with k number of features
#         end
            if (not ('coef0' is not None) ):
                coef0 = 0
            if (not ('cost' is not None) ):
                cost = 1
            if (not ('nu' is not None) ):
                nu = 0.5
            if (not ('epsilon_regr' is not None) ):
                epsilon_regr = 0.1
            if (not ('chache' is not None) ):
                chache = 100
            if (not ('epsilon' is not None) ):
                epsilon = 0.001
            if (not ('shrinking' is not None) ):
                shrinking = 1
            if (not ('probability_estimates' is not None) ):
                probability_estimates = 0
            if (not ('nfold' is not None) ):
                nfold = 5
            if (not ('weight' is not None) ):
                weight = 1
            if (not ('quiet' is not None) ):
                quiet = 1
            # Check if a variable is present in the structure, if so it means
# that the user has define it
            if (not isfield(options,'svm_type') ):
                options.svm_type = svm_type
            if (not isfield(options,'kernel_type') ):
                options.kernel_type = kernel_type
            if (not isfield(options,'kernel_degree') ):
                options.kernel_degree = kernel_degree
            #         if (~isfield(options, 'gamma'))
#             options.gamma = gamma;
#         end
            if (not isfield(options,'coef0') ):
                options.coef0 = coef0
            if (not isfield(options,'cost') ):
                options.cost = cost
            if (not isfield(options,'nu') ):
                options.nu = nu
            if (isfield(options,'epsilon_regr')):
                options.epsilon_regr = epsilon_regr
            if (not isfield(options,'chache') ):
                options.chache = chache
            if (not isfield(options,'epsilon') ):
                options.epsilon = epsilon
            if (isfield(options,'shrinking')):
                options.shrinking = shrinking
            if (not isfield(options,'probability_estimates') ):
                options.probability_estimates = probability_estimates
            if (not isfield(options,'nfold') ):
                options.nfold = nfold
            if (not isfield(options,'weight') ):
                options.weight = weight
            if (not isfield(options,'quiet') ):
                options.quiet = quiet
            cmd = options
    
    return cmd