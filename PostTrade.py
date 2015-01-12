# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import pandas as  pd
from ggplot import *

# <codecell>

d='/home/user1/Desktop/Share2Windows/QimbReports/'
reports = os.listdir(d)

# <codecell>

#import OO orders from reports
df = pd.DataFrame(columns=['Date','Symbol','Side','Price','Time','Filled'])
for f_name in reports:
    day=f_name.split('_')[1]
    file = open(d+f_name, 'r')
    lineInd=0
    dropped = False
    activated = False
    for line in file:      
        if dropped:
            continue
        
        words = line.split()
        
        if len(words)==1:
            if words[0]=='!':
                break
                
        if activated:
            words = line.split(';')
            if words[7]=='3':
                df=df.append({'Date':datetime.datetime.strptime(day,'%Y-%m-%d'),'Symbol':words[0],\
                              'Side':words[1],'Price':float(words[2]),\
                              'Time':words[10],'Filled':float(words[4])!=0},ignore_index=True)
            continue
        
        if lineInd==4:
            if words[2]!='Qimb':
                dropped = True
                
        if len(words)==1:
            if words[0]=='Orders:':
                activated = True           
            
        lineInd+=1
    file.close()
df = df.sort('Date')
df.index = range(df.shape[0])
df = df.ix[df.Filled.nonzero()[0][0]:,:]
df.index = range(df.shape[0])
for i in range(df.shape[0]):
    df.Symbol[i] = df.Symbol[i].strip()

# <codecell>

dates = list(set(df.Date))
dates.sort()

# <codecell>

#get Symbols where OO orders where sent
i=0
for d in dates:
    s = d.strftime('%Y-%m-%d')+' StockList='
    for symbol in set(df[df.Date==d].Symbol):
        s +='%s,' %  symbol.strip()
        i+=1
    print s 

# <codecell>

OOsent = pd.read_csv('/home/user1/Desktop/Share2Windows/OOfeatures.csv',header=None)
OOsent['Filled'] = 0
for d in dates:
    dfSymbolsSet = set(df[(df.Date==d) & (df.Filled)].Symbol)
    featuresSymbolsSet = set(OOsent[(OOsent[0]==d.strftime('%Y-%m-%d'))][1])
    filledOOorders = set.intersection(*[dfSymbolsSet,featuresSymbolsSet])
    
    for symbol in filledOOorders:
        OOsent.Filled[(OOsent[0]==d.strftime('%Y-%m-%d')) & (OOsent[1]==symbol)] = 1
OOsent['Date']=OOsent[0]

# <codecell>

X = OOsent.ix[:,2:-3]
y = OOsent.ix[:,-2]

# <codecell>

import pandas
import numpy
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestClassifier as RF
import numpy
import math
import qimbs

# <codecell>

#Train classifier
def RunModel(clf_class,X,y,test_size,n_folds,dates,datesDF):

    r = range(len(dates))

    labels =  numpy.sort(list(set(y)))
    test_cm = np.zeros((len(labels),len(labels)))
    train_cm = np.zeros((len(labels),len(labels)))
    
    if type(clf_class)!=str: 
        CLF_BEST = clf_class()
    TEST_F_Score = 0

    for i in range(n_folds): 
        np.random.shuffle(r)
        test_days = r[:test_size] 
        train_days = r[test_size:] 

        Xtrain = X.ix[datesDF.ix[train_days],:]
        Xtest = X.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]]        

        Xtrain.index = range(Xtrain.shape[0])
        Xtest.index = range(Xtest.shape[0])
        ytrain.index = range(ytrain.shape[0])
        ytest.index = range(ytest.shape[0])
        
        if (len((pd.isnull(Xtrain)).all(1).nonzero()[0])>0):
            nonanind = (~pd.isnull(Xtrain)).all(1).nonzero()[0]
            ytrain = ytrain[nonanind]
            Xtrain = Xtrain.ix[nonanind,:]
            Xtrain.index = range(Xtrain.shape[0])
            ytrain.index = range(ytrain.shape[0])
            
        if (len((pd.isnull(Xtest)).all(1).nonzero()[0])>0):
            nonanind = (~pd.isnull(Xtest)).all(1).nonzero()[0]
            ytest = ytest[nonanind]
            Xtest = Xtest.ix[nonanind,:]
            Xtest.index = range(Xtest.shape[0])
            ytest.index = range(ytest.shape[0]) 

        if (type(clf_class()) ==  type(LR())) :
            clf = clf_class(class_weight='auto',C=0.1)
        if (type(clf_class()) ==  type(SVC())):
            clf = clf_class(class_weight='auto',probability=True)
        if (type(clf_class()) ==  type(RF())):
            clf = clf_class(n_jobs=4,min_samples_split = Xtrain.shape[0]*0.05, \
                                       criterion = 'entropy', n_estimators = 10)
        if (type(clf_class()) ==  type(GBC())):
            clf = clf_class(min_samples_split = Xtrain.shape[0]*0.05,init='zero')
        if (type(clf_class()) ==  type(GBR())):
            clf = clf_class(init='zero')

        clf.fit(Xtrain,ytrain)


        if (type(clf_class()) !=  type(GBR())):
            probaTest = clf.predict_proba(Xtest).astype(float)
            probaTrain = clf.predict_proba(Xtrain).astype(float)

            ypred = clf.classes_[numpy.argmax(probaTest,axis=1)]
            ypredTrain = clf.classes_[numpy.argmax(probaTrain,axis=1)]  

            test_cm_tmp = confusion_matrix(ytest,ypred,labels).astype(float)
            test_cm += test_cm_tmp/n_folds
            train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds

        else:
            probaTest = clf.predict(Xtest).astype(float)
            probaTrain = clf.predict(Xtrain).astype(float)

            ypred = probaTest>0.5
            ypredTrain = probaTrain>0.5  

            test_cm_tmp = confusion_matrix(ytest,ypred,labels).astype(float)
            test_cm += test_cm_tmp/n_folds
            train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds     
            
        pr =  qimbs.Precision_Recall(test_cm_tmp, labels)
        if (pr[2]>TEST_F_Score):
            TEST_F_Score =  pr[2]
            CLF_BEST = clf  
    
    #print TEST_ERRORS
    print "max F_Score ",TEST_F_Score

    test_pr = qimbs.Precision_Recall(test_cm, labels)
    train_pr = qimbs.Precision_Recall(train_cm, labels)

    return test_cm,CLF_BEST

# <codecell>

cm,clf = RunModel(RF,X,y,int(0.15*len(dates)),20,dates,qimbs.dates_tmp_df(OOsent))
#cm = RunModel(RF,X[fi['Feature'][:10]],y,0.1,10)
pr = qimbs.Precision_Recall(cm, labels)
fig1 = plt.figure(figsize=(15, 5))
plt.clf()
ax1 = fig1.add_subplot(1,3,1)

qimbs.draw_confusion_matrix(cm,  numpy.sort(list(set(y))), fig1, ax1)
print 'Precision - %s, Recall - %s, F_Score - %s' % (pr[0],pr[1],pr[2])

clf.fit(X,y)
proba = clf.predict_proba(X).astype(float)
ypred = clf.classes_[numpy.argmax(proba,axis=1)]
cm_new = confusion_matrix(y,ypred,labels).astype(float)

# <codecell>

fig1 = plt.figure(figsize=(15, 5))
plt.clf()
ax1 = fig1.add_subplot(1,3,1)
qimbs.draw_confusion_matrix(cm_new,  numpy.sort(list(set(y))), fig1, ax1)
pr = qimbs.Precision_Recall(cm_new, labels)
print 'Precision - %s, Recall - %s, F_Score - %s' % (pr[0],pr[1],pr[2])

# <codecell>

fi = pd.DataFrame()
fi['Feature'] = list(X.columns)
fi['Impotrance'] = clf.feature_importances_
fi=fi.sort(columns=['Impotrance'],ascending=False)
fi['Index'] = range(X.shape[1])
fi.index = fi['Index']

ggplot(fi,aes('Index','Impotrance',label='Feature')) +\
geom_point() + geom_text(vjust=0.005)

# <codecell>

#import stocks pnl
df = pd.DataFrame(columns=['Symbol','Shares','Pnl'])
for f_name in reports:
    day=f_name.split('_')[1]
    file = open(d+f_name, 'r')
    lineInd=0
    dropped = False
    activated = False
    for line in file:      
        if dropped:
            continue
        
        words = line.split()
        
        if len(words)==1:
            if words[0]=='!':
                break
                
        if len(words)==1:
            if words[0]=='Orders:':
                break    
                
        if activated:
            
            words = line.split(';')
            
            if len(words)==1:
                break
                
            if len(words)==10:
                df=df.append({'Symbol':words[0].strip(),'Shares':0.5*float(words[2]),'Pnl':float(words[5])},ignore_index=True)
            continue
        
        if lineInd==4:
            if words[2]!='Qimb':
                dropped = True
                
        if len(words)==1:
            if words[0]=='Positions:':
                activated = True           
            
        lineInd+=1
    file.close()
    
df = df.groupby(df.Symbol).sum()
df['Pps'] = df.Pnl/df.Shares
df=df.sort('Pps')
df.Pps[abs(df.Pps)>0.2].plot()

