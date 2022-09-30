import numpy as np
    
def calcError(trueLabelling = None,segLabelling = None,labels = None): 
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
    
    nrX,nrY = trueLabelling.shape
    totNrPixels = nrX * nrY
    nrPixelsPerClass = np.transpose(np.zeros((1,len(labels))))
    nrClasses = len(labels)
    errorMatrix = np.zeros((len(labels),len(labels)))
    errorMatrixPerc = np.zeros((len(labels),len(labels)))
    for l_true in np.arange(1,len(labels)+1).reshape(-1):
        tmp_true = find(trueLabelling == (l_true - 1))
        nrPixelsPerClass[l_true] = len(tmp_true)
        for l_seg in np.arange(1,len(labels)+1).reshape(-1):
            tmp_seg = find(segLabelling == (l_seg - 1))
            nrPixels = len(intersect(tmp_true,tmp_seg))
            errorMatrix[l_true,l_seg] = nrPixels
    
    # classWeight = nrPixelsPerClass/totNrPixels;
    diagVector = diag(errorMatrix)
    class_accuracy = (diagVector / (nrPixelsPerClass))
    average_accuracy = mean(class_accuracy)
    overall_accuracy = sum(segLabelling == trueLabelling) / len(trueLabelling)
    kappa_accuracy = (sum(errorMatrix) * sum(diag(errorMatrix)) - sum(errorMatrix) * np.sum(errorMatrix, 2-1)) / (sum(errorMatrix) ** 2 - sum(errorMatrix) * np.sum(errorMatrix, 2-1))
    return overall_accuracy,kappa_accuracy,average_accuracy,class_accuracy,errorMatrix