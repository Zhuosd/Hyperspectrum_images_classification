import numpy as np
    
def getPatterns(varargin = None): 
    # (data, label, classActive)
    
    # From a label image (e.g., training or test image) extract the features of
# the labelled patterns generating a 'traindata' data structure (matrix of
# Nsamples-by-Nfeatures+1, having as last column the labelled values).
    
    if 2 == len(varargin):
        data = varargin[0]
        label_img = varargin[2]
        Nclasses = np.amax(label_img)
        isClassActive = np.ones((1,Nclasses))
    else:
        if 3 == len(varargin):
            data = varargin[0]
            label_img = varargin[2]
            isClassActive = varargin[3]
            Nclasses = np.amax(label_img)
            if isClassActive.shape[2-1] != Nclasses:
                raise Exception('ClassActive elements does not match the number of classes\n')
        else:
            raise Exception('Wrong number of inputs\n')
    
    ClassActiveIdx = find(isClassActive)
    NclassesActive = len(find(isClassActive))
    Nsamples = 0
    Nfeats = data.shape[3-1]
    for i in np.arange(1,NclassesActive+1).reshape(-1):
        NelemPerClass[i] = len(find(label_img == ClassActiveIdx(i)))
        Nsamples = Nsamples + NelemPerClass(i)
    
    # Priors = NelemPerClass./Nsamples;
    
    Pat = np.zeros((Nfeats,Nsamples))
    Plabel = np.zeros((1,Nsamples))
    pos = 0
    for i in np.arange(1,NclassesActive+1).reshape(-1):
        r,c = find(label_img == i)
        NelemPerClass[i] = len(r)
        for j in np.arange(1,NelemPerClass(i)+1).reshape(-1):
            Pat[:,pos + j] = np.squeeze(data(r(j),c(j),:))
        Plabel[np.arange[pos + 1,pos + NelemPerClass[i]+1]] = ClassActiveIdx(i)
        pos = pos + NelemPerClass(i)
    
    if nargout > 2:
        Ximg = reshape(data,[],data.shape[3-1])
    
    return Pat,Plabel,Ximg