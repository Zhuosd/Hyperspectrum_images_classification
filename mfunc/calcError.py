from cmath import e
import numpy as np

    
def calcError(trueLabelling, segLabelling, labels): 
    # calculates square array of numbers organized in rows and columns which express the
# percentage of pixels assigned to a particular category (in segLabelling) relative
# to the actual category as indicated by reference data (trueLabelling)
# errorMatrix(i,j) = nr of pixels that are of class i-1 and were
# classified as class j-1
# accuracy is essentially a measure of how many ground truth pixels were classified
# correctly (in percentage).
# average accuracy is the average of the accuracies for each class
# overall accuracy is the accuracy of each class weighted by the proportion
# of test samples for that class in the total training set
    
    nrX, nrY = trueLabelling.shape
    totNrPixels = nrX * nrY
    nrPixelsPerClass = np.transpose(np.zeros((1,len(labels))))
    nrClasses = len(labels)
    errorMatrix = np.zeros((len(labels),len(labels)))
    errorMatrixPerc = np.zeros((len(labels),len(labels)))
    for l_true in np.arange(len(labels)):
        # print('l_true', l_true)
        # tmp_true = find(trueLabelling == (l_true - 1))
        # temp_equ = trueLabelling == (l_true - 1)
        tmp_true = (trueLabelling == l_true).nonzero()
        # print('tmp_true  -- ', tmp_true)
        # print('tmp_true  -- ', len(tmp_true[0]))
        # print('tmp_true  -- ', len(tmp_true[1]))
        nrPixelsPerClass[l_true] = len(tmp_true[0])
        for l_seg in np.arange(len(labels)):
            # tmp_seg = find(segLabelling == (l_seg - 1))
            tmp_seg = (segLabelling == (l_seg)).nonzero()
            # nrPixels = len(intersect(tmp_true,tmp_seg))
            # np.intersect1d(tmp_true)
            # nrPixels_s = tmp_true.instersetion(tmp_seg)
            nrPixels_s = np.intersect1d(tmp_true, tmp_seg)
            nrPixels = len(nrPixels_s)
            errorMatrix[l_true,l_seg] = nrPixels
    

    # classWeight = nrPixelsPerClass/totNrPixels;
    diagVector = np.diag(errorMatrix)
    class_accuracy = diagVector / nrPixelsPerClass.T
    average_accuracy = np.mean(class_accuracy)
    overall_accuracy = np.sum(np.equal(segLabelling, trueLabelling)) / len(trueLabelling[0])
    # np.sum(M, axis=0) axis=0 对列进行求和， axis=1 对每行进行求和
    kappa_accuracy = (np.sum(errorMatrix) * np.sum(np.diag(errorMatrix)) - np.sum(errorMatrix) * np.sum(errorMatrix, axis=1)) / (np.sum(errorMatrix) ** 2 - np.sum(errorMatrix) * np.sum(errorMatrix, axis=1))      # np.sum(M, axis=0) axis=0 对列进行求和， axis=1 对每行进行求和
    return overall_accuracy, kappa_accuracy, average_accuracy, class_accuracy, errorMatrix
