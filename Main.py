import pandas as pd
from os import listdir
import numpy
from sklearn.model_selection import StratifiedKFold
from FS_ALO import WFS
from FW_ALO import WFW
from WFSWFW_ALO import WFSWFW
import matplotlib.pyplot as plt

def main_CV():
    path='./datasets'
    direc=sorted(listdir(path))
    print(direc)
    population=20
    Total_iter=200
    total_reruns=20
    Cost=numpy.zeros([len(direc),total_reruns,3],dtype=numpy.float64)
    CC=numpy.zeros([len(direc),total_reruns,Total_iter,3],dtype=numpy.float64)
    Best_WFS=[]
    Best_WFW=[]
    Best_WFSWFW=[]
    for dir_idx in range(0,len(direc)):
        data_path=path+'/'+direc[dir_idx]
        print(data_path)
        tl=pd.read_csv(data_path,header=None,dtype='str')
        tl.drop([0],axis=0,inplace=True)
        data_array=numpy.array(tl)
        data=data_array.astype(numpy.float)
        dim=data.shape
        train=data[:,0:dim[1]-1]
        label=data[:,dim[1]-1]
        for i in range(0,total_reruns):
            cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
            Cost[dir_idx,i,0],Elite_pos1,CC[dir_idx,i,:,0]=WFS(train,label,cv,population,Total_iter)
            Cost[dir_idx,i,1],Elite_pos2,CC[dir_idx,i,:,1]=WFW(train,label,cv,population,Total_iter)
            Cost[dir_idx,i,2],FS_pos,FW_pos,CC[dir_idx,i,:,2]=WFSWFW(train,label,cv,population,Total_iter)
            Best_WFS.append(Elite_pos1)
            Best_WFW.append(Elite_pos2)
            Best_WFSWFW.append(FS_pos)
            Best_WFSWFW.append(FW_pos)
        
    mean_CC=numpy.mean(CC,axis=1)
    for i in range(0,1):
        plt.plot(mean_CC[i,:,0],color='r')
        plt.plot(mean_CC[i,:,1],color='b')
        plt.plot(mean_CC[i,:,2],color='g')
    return Cost,Best_WFS,Best_WFW,Best_WFSWFW,CC

# Main program
Cost,Best_WFS,Best_WFW,Best_WFSWFW,CC=main_CV()

