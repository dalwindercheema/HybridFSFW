import numpy
from FS_ALO import WFS
from FW_ALO import WFW

def select_feat(train,feat_weights):
    dim=train.shape[1]
    total_feat=numpy.count_nonzero(feat_weights)
    ntrain=numpy.zeros((train.shape[0],total_feat))
    cont=0
    for i in range(0,dim):
        if(feat_weights[i]!=0):
            ntrain[:,cont]=train[:,i]
            cont=cont+1
    return ntrain

def WFSWFW(train,label,cv,P,Iter):
    half_iter=int(Iter/2)
    FSCost,FSElite_pos,FSCC=WFS(train,label,cv,P,half_iter)
    ntrain=select_feat(train,FSElite_pos)
    FWCost,FWElite_pos,FWCC=WFW(ntrain,label,cv,P,Iter-half_iter)
    CC=numpy.append(FSCC,FWCC)
    return FWCost,FSElite_pos,FWElite_pos,CC
    
    