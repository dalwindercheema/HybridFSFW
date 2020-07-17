import pandas as pd
from os import listdir
import numpy
from sklearn.model_selection import StratifiedKFold
from FW_ALO import ALO_KNN

def main_CV():
    path='./datasets'
    direc=sorted(listdir(path))
    print(direc)
    population=20
    Total_iter=200
    total_reruns=10
    Cost=numpy.zeros([len(direc),total_reruns],dtype=numpy.float64)
    CC=numpy.zeros([len(direc),total_reruns,Total_iter],dtype=numpy.float64)
    Best_sol=[]
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
            Cost[dir_idx,i],Elite_pos,CC[dir_idx,i,:]=ALO_KNN(train,label,cv,population,Total_iter)
            Best_sol.append(Elite_pos)
    return Cost,Best_sol,CC

# Main program
Cost,Best_sol,CC=main_CV()

