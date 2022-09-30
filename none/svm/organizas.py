import numpy as np
    
def organizas_tr(vectTR = None): 
    classes = np.transpose(setdiff(unique(vectTR),0))
    nx,ny = vectTR.shape
    # vect_TR_3=zeros(nx,ny,classes);
    vect_TR_3 = np.zeros((nx,ny,np.amax(classes)))
    for valor in classes.reshape(-1):
        for i in np.arange(1,nx+1).reshape(-1):
            for j in np.arange(1,ny+1).reshape(-1):
                if vectTR(i,j) != valor:
                    vect_TR_3[i,j,valor] = 0
                else:
                    vect_TR_3[i,j,valor] = valor
    
    return vect_TR_3,classes