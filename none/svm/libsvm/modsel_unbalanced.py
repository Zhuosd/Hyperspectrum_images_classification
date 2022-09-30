import numpy as np
import matplotlib.pyplot as plt
    
def modsel_unbalanced(label = None,inst = None): 
    # Model selection for (lib)SVM by searching for the best param on a 2D grid
# example:
    
    # load heart_scale.mat
# [bestc,bestg,bestcv,hC] = modsel(heart_scale_label,heart_scale_inst);
    
    #contour plot
    wa = lambda e1 = None,e2 = None,s1 = None,s2 = None: (1 - (e1 * 1 / s1 + e2 * 1 / s2) / 2) * 100
    
    label[label == - 1] = 2
    label[label == 0] = 2
    fold = 10
    c_begin = - 5
    c_end = 10
    c_step = 1
    g_begin = 8
    g_end = - 8
    g_step = - 1
    #c_begin = 0; c_end = 8; c_step = 1;
#g_begin = -4; g_end = -12; g_step = -1;
    bestcv = 0
    bestc = 2 ** c_begin
    bestg = 2 ** g_begin
    # Preallocation of memory -> Just for speed
    Z = np.zeros((len(np.arange(c_begin,c_end+c_step,c_step)),len(np.arange(g_begin,g_end+g_step,g_step))))
    i = 1
    j = 1
    indices = crossvalind('Kfold',label,fold)
    for log2c in np.arange(c_begin,c_end+c_step,c_step).reshape(-1):
        for log2g in np.arange(g_begin,g_end+g_step,g_step).reshape(-1):
            cmd = np.array(['-w1 5 -w2 1 -c ',num2str(2 ** log2c),' -g ',num2str(2 ** log2g)])
            cp = classperf(label)
            for k in np.arange(1,fold+1).reshape(-1):
                test = (indices == k)
                train = not test 
                mdl = svmtrain(label(train,:),inst(train,:),cmd)
                class_ = svmpredict(label(test,:),inst(test,:),mdl)
                classperf(cp,class_,test)
            cv = wa(cp.errorDistributionByClass(1),cp.errorDistributionByClass(2),cp.SampleDistributionByClass(1),cp.SampleDistributionByClass(2))
            if (cv > bestcv) or ((cv == bestcv) and (2 ** log2c < bestc) and (2 ** log2g == bestg)):
                bestcv = cv
                bestc = 2 ** log2c
                bestg = 2 ** log2g
            print(np.array([num2str(log2c),' ',num2str(log2g),' (best c=',num2str(bestc),' g=',num2str(bestg),' rate=',num2str(bestcv),'%)']))
            Z[i,j] = cv
            j = j + 1
        j = 1
        i = i + 1
    
    xlin = np.linspace(c_begin,c_end,Z.shape[1-1])
    ylin = np.linspace(g_begin,g_end,Z.shape[2-1])
    X,Y = np.meshgrid(xlin,ylin)
    acc_range = (np.arange(np.ceil(bestcv) - 3.5,np.ceil(bestcv)+0.5,0.5))
    C,hC = plt.contour(X,Y,np.transpose(Z),acc_range)
    #legend plot
    set(get(get(hC,'Annotation'),'LegendInformation'),'IconDisplayStyle','Children')
    ch = get(hC,'Children')
    tmp = cell2mat(get(ch,'UserData'))
    M,N = unique(tmp)
    c = setxor(N,np.arange(1,len(tmp)+1))
    for i in np.arange(1,len(N)+1).reshape(-1):
        set(ch(N(i)),'DisplayName',num2str(acc_range(i)))
    
    for i in np.arange(1,len(c)+1).reshape(-1):
        set(get(get(ch(c(i)),'Annotation'),'LegendInformation'),'IconDisplayStyle','Off')
    
    plt.legend('show')
    #bullseye plot
    hold('on')
    plt.plot(log2(bestc),log2(bestg),'o','Color',np.array([0,0.5,0]),'LineWidth',2,'MarkerSize',15)
    axs = get(gca)
    plt.plot(np.array([axs.XLim(1),axs.XLim(2)]),np.array([log2(bestg),log2(bestg)]),'Color',np.array([0,0.5,0]),'LineStyle',':')
    plt.plot(np.array([log2(bestc),log2(bestc)]),np.array([axs.YLim(1),axs.YLim(2)]),'Color',np.array([0,0.5,0]),'LineStyle',':')
    hold('off')
    plt.title(np.array([[np.array(['Best log2(C) = ',num2str(log2(bestc)),',  log2(gamma) = ',num2str(log2(bestg)),',  Accuracy = ',num2str(bestcv),'%'])],[np.array(['(C = ',num2str(bestc),',  gamma = ',num2str(bestg),')'])]]))
    plt.xlabel('log2(C)')
    plt.ylabel('log2(gamma)')
    return bestc,bestg,bestcv,hC