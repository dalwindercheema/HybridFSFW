import numpy
from sklearn import neighbors
from sklearn.metrics import accuracy_score

def initialization(lb,ub,P):
    dim=lb.shape[0]
    pos=numpy.zeros((P,dim))
    for i in range(0,dim):
        pos[:,i]=numpy.random.uniform(lb[i],ub[i],P)
    return pos

def sort_pos(fitness,solutions):
    sort_index = numpy.argsort(fitness)
    P,dim=solutions.shape
    npos=numpy.zeros((P,dim))
    for i in range(0,P):
        npos[i,:]=solutions[int(sort_index[i]),:]
    return npos

def sort_fit(fitness):
    sort_index = numpy.argsort(fitness)
    P=fitness.shape[0]
    ncost=numpy.zeros(P)
    for i in range(0,P):
        ncost[i]=fitness[int(sort_index[i])]
    return ncost

def RWSelection(acc,P):
    acc=[1/x for x in acc]
    cumulative_sum=numpy.cumsum(acc)
    p=numpy.random.rand()*cumulative_sum[P-1]
    idx=-1
    for i in range(0,P):
        if(cumulative_sum[i]>p):
            idx=i
            break
    return idx

def calc_weights(train,feat_weights):
    dim=train.shape[1]
    total_feat=numpy.count_nonzero(feat_weights)
    ntrain=numpy.zeros((train.shape[0],total_feat))
    cont=0
    for i in range(0,dim):
        if(feat_weights[i]!=0):
            ntrain[:,cont]=train[:,i]*feat_weights[i]
            cont=cont+1
    return ntrain

def KNNCV(train,label,cv,K):
    accuracy=[]
    dim=train.shape[1]
    model = neighbors.KNeighborsClassifier(int(K),weights='uniform')
    for train_index, test_index in cv.split(train,label):
        train_data=train[train_index,0:dim]
        train_label=label[train_index]
        test_data=train[test_index,0:dim]
        test_label=label[test_index]
        model.fit(train_data,train_label)
        pred_label = model.predict(test_data)
        acc=accuracy_score(test_label,pred_label)
        accuracy.append(acc)
    return numpy.mean(accuracy)

def get_error(train,label,cv,kval,solution):
    weighted_data=calc_weights(train,solution)
    total_feat=weighted_data.shape[1]
    if(total_feat==0):
        fitness=1
    else:
        fitness=1-KNNCV(weighted_data,label,cv,kval)
    return fitness

def RW(dim,lb,ub,Iter,current_iter,AL):
    I=1
    ratio=current_iter/Iter
    if (current_iter>(Iter*0.95)):
        I=1+50*ratio
    elif (current_iter>(Iter*0.90)):
        I=1+20*ratio
    elif (current_iter>(Iter*0.75)):
        I=1+10*ratio
    elif (current_iter>(Iter*0.50)):
        I=1+5*ratio
    
    lb=lb/I
    ub=ub/I
    if(numpy.random.rand()<0.5):
        lb=lb+AL
    else:
        lb=-lb+AL
    
    if(numpy.random.rand()>0.5):
        ub=ub+AL
    else:
        ub=-ub+AL
    RW=[]
    for i in range(0,dim):
        X=numpy.cumsum(2*(numpy.random.rand(Iter,1)>0.5)-1)
        a=min(X)
        b=max(X)
        c=lb[i]
        d=ub[i]
        tmp1=(X-a)/(b-a)
        X_norm=c+tmp1*(d-c)
        RW.append(X_norm)
    return numpy.array(RW)

def blx_oper(RW,E,lb,ub):
    dim=RW.shape
    ant_pos=numpy.zeros(dim)
    d=abs(RW-E)
    for j in range(0,dim[0]):
        x1=min(RW[j],E[j])-0.5*d[j]
        x2=max(RW[j],E[j])+0.5*d[j]
        rd=numpy.random.uniform(x1,x2,1)
        ant_pos[j]=max(min(rd,ub[j]),lb[j])
    return ant_pos

def ALO_KNN(train,label,cv,P,Iter):
    size=train.shape
    dim=size[1]
    lb=numpy.zeros(dim)
    ub=numpy.ones(dim)
    lb=numpy.append(lb,1)
    ub=numpy.append(ub,10)
    CC=numpy.zeros(Iter)
    antlions=initialization(lb,ub,P)
    alion_acc=numpy.zeros(P)
    for i in range(0,P):
        antlions[i,dim]=numpy.rint(antlions[i,dim])
        alion_acc[i]=get_error(train,label,cv,antlions[i,dim],antlions[i,0:dim])
    
    sorted_antlions=sort_pos(alion_acc,antlions)
    alion_acc=sort_fit(alion_acc)
    
    Elite_acc=numpy.copy(alion_acc[0])
    Elite_pos=numpy.copy(sorted_antlions[0,:])    

    CC[0]=Elite_acc
    current_iter=2
    while current_iter<=Iter:
        print(current_iter)
        ant_pos=numpy.zeros([P,dim+1])
        ant_acc=numpy.zeros(P)
        for i in range(0,P):
            idx=RWSelection(alion_acc,P)
            if(idx==-1):
                idx=0
            RW_EL=RW(dim+1,lb,ub,Iter,current_iter,Elite_pos)
            RW_RWS=RW(dim+1,lb,ub,Iter,current_iter,sorted_antlions[idx,:])
            cont_feat=blx_oper(RW_EL[:,current_iter-1],RW_RWS[:,current_iter-1],lb,ub)
            ant_pos[i,0:dim]=cont_feat[0:dim]
            ant_pos[i,dim]=numpy.rint(cont_feat[dim])
            ant_acc[i]=get_error(train,label,cv,ant_pos[i,dim],ant_pos[i,0:dim])
            
        total_fitness=numpy.append(alion_acc,ant_acc)
        total_pos=numpy.concatenate((sorted_antlions,ant_pos),axis=0)
        
        tmp_sorted_antlions=sort_pos(total_fitness,total_pos)
        total_fitness=sort_fit(total_fitness)
        
        alion_acc=numpy.copy(total_fitness[0:P])
        sorted_antlions=numpy.copy(tmp_sorted_antlions[0:P,:])
        
        Elite_acc=numpy.copy(alion_acc[0])
        Elite_pos=numpy.copy(sorted_antlions[0,:])
        CC[current_iter-1]=Elite_acc
        current_iter=current_iter+1
    return Elite_acc,Elite_pos,CC