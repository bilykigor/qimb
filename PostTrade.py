# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os

import numpy
import pandas as  pd
import math

from ggplot import *

import qimbs
import mmll
from EnsembleClassifier import EnsembleClassifier
from EnsembleClassifier import LabeledEstimator

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import confusion_matrix

# <codecell>

reload(mmll)

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
float(sum(y))/len(y)

# <codecell>

reload(qimbs)
reload(mmll)

# <codecell>

reload(EnsembleClassifier)

# <codecell>

lclf=EnsembleClassifier.LabeledEstimator(RF)
labels = np.ones(len(y))
labels[:300]=0
lclf.fit(X,y,labels = labels)
for value in lclf.clfs.itervalues():
    print value

# <codecell>

lclf.predict_proba(X)

# <codecell>

eclf = EnsembleClassifier(
clfs=[
RF( min_samples_split = len(y)*0.03,criterion='entropy',n_jobs=4)
#GBC(min_samples_split = len(y)*0.03,init='zero'),
#LR(class_weight='auto',C=0.1)
])

# <codecell>

cm,clf = mmll.clf_cross_validation(eclf,X,y,test_size=20,n_folds=10,labels = OOsent.Date)

labels = numpy.sort(list(set(y)))

fig1 = plt.figure(figsize=(15, 5))
ax1 = fig1.add_subplot(1,3,1)
mmll.draw_confusion_matrix(cm, labels , fig1, ax1)

#clf.fit(X,y)
#proba = clf.predict_proba(X).astype(float)
#ypred = clf.classes_[numpy.argmax(proba,axis=1)]
#cm_new = confusion_matrix(y,ypred,labels).astype(float)

# <codecell>

cm

# <codecell>

fi = pd.DataFrame()
fi['Feature'] = list(X.columns)
fi['Impotrance'] = clf.clfs[0].feature_importances_
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

