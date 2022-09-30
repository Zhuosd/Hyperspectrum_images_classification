## Cleanup
import numpy as np
clear('all')
close_('all')
## Adding Paths LIBSVM Matlab
addpath('D:\MAHDI\Code\libsvm-3.20')
addpath('D:\MAHDI\Code\libsvm-3.20\matlab')
addpath('D:\MAHDI\Code\libsvm-3.20\windows')
## Example on Heart Scale Data
heart_scale_label,heart_scale_inst = libsvmread('D:\MAHDI\Code\libsvm-3.20\heart_scale')
# Train and Test Data Selection
N = 150

M = heart_scale_label.shape[1-1]

train_data = heart_scale_inst(np.arange(1,N+1),:)
train_label = heart_scale_label(np.arange(1,N+1),:)
test_data = heart_scale_inst(np.arange(N + 1,270+1),:)
test_label = heart_scale_label(np.arange(N + 1,270+1),:)
# Linear Kernel
model_linear = svmtrain(train_label,train_data,'-t 0')
model_precomputed = svmtrain(train_label,np.array([np.transpose((np.arange(1,N+1))),train_data * np.transpose(train_data)]),'-t 4')
# Applying SVM
predict_label_L,accuracy_L,dec_values_L = svmpredict(test_label,test_data,model_linear)
predict_label_P,accuracy_P,dec_values_P = svmpredict(test_label,np.array([np.transpose((np.arange(1,M - N+1))),test_data * np.transpose(train_data)]),model_precomputed)