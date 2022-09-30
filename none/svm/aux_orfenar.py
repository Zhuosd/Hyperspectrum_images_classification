import numpy as np
    
def aux_ordenar_v4(vect_TR = None,vectProb = None,no_lines = None,no_col = None): 
    vect_clases = []
    ## Organizar las clases segun el numero de pixeles
    
    vect_TR_3,classes = organizas_tr(vect_TR)
    x,y,b = vect_TR_3.shape
    vect_TR_3_2 = reshape(vect_TR_3,x * y,b)
    for i in classes.reshape(-1):
        vect_clases = np.array([vect_clases,sum((vect_TR_3_2(:,i) > 0))])
    
    valor,p_tr = __builtint__.sorted(vect_clases,'descend')
    #p_tr= orden de las clases
    
    ## Leer el vector de probabilidad (valor, posicion) (21025*16)
    ns,p = vectProb.shape
    ## nuevas variables
    M = []
    order = []
    ordervalue = []
    ## comparamos todas las clases (con el orden establecido) con los mapas de
## abundancia
    jset = np.arange(1,p+1)
    for c in p_tr.reshape(-1):
        pos = find(vect_TR == c)
        aux = []
        for j in jset.reshape(-1):
            a = (vectProb(pos,j))
            average = prctile(a,50)
            aux = np.array([aux,average])
        #          plot (aux)
        v,pp = np.amax(aux)
        pp = jset(pp)
        order = np.array([order,pp])
        jset = setdiff(jset,order)
        ordervalue = np.array([ordervalue,v])
        print(' training class # %3.0d -> map %3.0d prob. \n' % (c,pp))
    
    v,pp = __builtint__.sorted(p_tr)
    order = order(pp)
    # M = [M;[order,mean(ordervalue)]];
# [v,ppp]=max(M(:,end));
# v
    
    # order = M(ppp,1:end-1);
    vectProbOrdered = vectProb(:,order)
    # visualize results
    mapTR = vect_TR
    mapProb = reshape(vectProb,no_lines,no_col,p)
    temp = []
    # # # Pintar mapas
# # for i=1:p
# #   #  temp = [temp;[mapTR==i,mapProb(:,:,order(i))]];
# #
# #     imagesc([mapTR==i,mapProb(:,:,order(i))]);figure
# # end
# #     close
    return vectProbOrdered,order,ordervalue