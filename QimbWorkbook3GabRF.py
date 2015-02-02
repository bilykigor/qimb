# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import mmll
import EnsembleClassifier as ec
import mio

import numpy as np
import pandas as pd
from math import factorial
from ggplot import *

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.base import clone

# <codecell>

reload(qimbs)
reload(mmll)

# <codecell>

#Procees c++ created features
f = '/home/user1/Desktop/Share2Windows/Qimb/firstImbNoImbS.csv' 
                   
imbalanceMsg = pd.read_csv(f, low_memory=False)

#imbalanceMsg = imbalanceMsg[imbalanceMsg.ImbInd!=0]
#imbalanceMsg.index = range(imbalanceMsg.shape[0])

print imbalanceMsg.shape

imbalanceMsg.head()

# <codecell>

sum(map(int,imbalanceMsg.AvgTradePrice==0))

# <codecell>

#Creating features
reload(qimbs)
fdf = qimbs.create_features34(imbalanceMsg)
fdf.head()

# <codecell>

list(fdf.columns)

# <codecell>

X = fdf.copy()
X=X[['Ask','AskD','Spread','D5', 'D555', 'D66', 'a1','a4','a5','Bid','BidD', 'D4', 'D444','ATP_P']]
X.head()

# <codecell>

X_pos = fdf.copy()
X_pos=X_pos[[
 'Ask',
 'D555',
 'AskD',
 'D5']]

X_neg = fdf.copy()
X_neg=X_neg[[
 'D4',
 'Bid',
 'D444',
 'BidD']]

ypln_pos = fdf.OPC_P/fdf.Ask_P-1
ypln_neg = 1-fdf.OPC_P/fdf.Bid_P

# <codecell>

ret = 0.05

ym_pos = fdf.imbInd.copy()
ym_pos[:] = 0
ym_pos[fdf.OPC_P-fdf.Ask_P>ret] = 1
ym_pos=ym_pos.reset_index(drop=True)

ym_neg = fdf.imbInd.copy()
ym_neg[:] = 0
ym_neg[fdf.OPC_P-fdf.Bid_P<-ret] = 1
ym_neg=ym_neg.reset_index(drop=True)

# <codecell>

ret = 0.01/100

yr_pos = fdf.imbInd.copy()
yr_pos[:] = 0
yr_pos[fdf.OPC_P/fdf.Ask_P-1.0>ret]=1

yr_neg = fdf.imbInd.copy()
yr_neg[:] = 0
yr_neg[fdf.OPC_P/fdf.Bid_P-1<-ret]=1

# <codecell>

(n,b,p) = matplotlib.pyplot.hist(list(ym_neg),bins=10,range=[0,1])

# <codecell>

print ym_pos.sum()
print ym_neg.sum()

# <codecell>

eclf_pos = RF(min_samples_split = int(len(yr_pos)*0.03),criterion='entropy',n_jobs=4)
'''ec.EnsembleClassifier(
clfs=[
RF(min_samples_split = int(len(yr_pos)*0.03),criterion='entropy',n_jobs=4)
#,GBC(min_samples_split = len(yr_pos)*0.03,init='zero')
#,LR(class_weight='auto',C=0.1)
])'''

eclf_neg = RF(min_samples_split = int(len(yr_neg)*0.03),criterion='entropy',n_jobs=4)
'''= ec.EnsembleClassifier(
clfs=[
RF(min_samples_split = int(len(yr_neg)*0.03),criterion='entropy',n_jobs=4)
#,GBC(min_samples_split = len(yr_neg)*0.03,init='zero')
#,LR(class_weight='auto',C=0.1)
])'''

# <codecell>

ttlabels = fdf.Date

# <codecell>

reload(qimbs)
reload(mmll)
reload(ec)

# <codecell>

cm,clf = mmll.clf_cross_validation(eclf_pos,X_pos,yr_pos,test_size=5,n_folds=50,train_test_labels = ttlabels,verbose = True)

# <codecell>

fi = pd.DataFrame()
fi['Feature'] = list(X_pos.columns)
fi['Impotrance'] = clf.feature_importances_
fi=fi.sort(columns=['Impotrance'],ascending=False)
fi['Index'] = range(X_pos.shape[1])
fi.index = fi['Index']
   
ggplot(fi,aes('Index','Impotrance',label='Feature')) +\
geom_point() + geom_text(vjust=0.005)

# <codecell>

cm,clf = mmll.clf_cross_validation(eclf_neg,X_neg,yr_neg,test_size=5,n_folds=50,train_test_labels = ttlabels,verbose = True)

# <codecell>

fi = pd.DataFrame()
fi['Feature'] = list(X_neg.columns)
fi['Impotrance'] = clf.feature_importances_
fi=fi.sort(columns=['Impotrance'],ascending=False)
fi['Index'] = range(X_neg.shape[1])
fi.index = fi['Index']
   
ggplot(fi,aes('Index','Impotrance',label='Feature')) +\
geom_point() + geom_text(vjust=0.005)

# <codecell>







# <codecell>






#Save configuration for C++

# <codecell>

def Forest2SqlZeroImbPerc(eclf_pos,eclf_neg,advantage,db,df):  
   
    X_p = df[['Ask','D555','AskD','D5']].copy()
    X_n = df[['Bid','D444','BidD','D4']].copy()   
    
    
    y_p = df.imbInd.copy()
    y_p[:] = 0
    if advantage>0:
        y_p[df.OPC_P/df.Ask_P-1.0>advantage/10000]=1 #y_p[df.OPC_P/df.Ref_P-1.0>advantage/10000]=1
    else:
        y_p[df.OPC_P/df.Ask_P-1.0<advantage/10000]=1

    y_n = df.imbInd.copy()
    y_n[:] = 0
    if advantage>0:
        y_n[df.OPC_P/df.Bid_P-1<-advantage/10000]=1#y_n[df.OPC_P/df.Ref_P-1<-advantage/10000]=1
    else:
        y_n[df.OPC_P/df.Bid_P-1>-advantage/10000]=1   
    
    ttlabels = df.Date.copy()
    
    cm,clf_pos = mmll.clf_cross_validation\
    (eclf_pos,X_p,y_p,test_size=20,n_folds=20,train_test_labels = ttlabels,verbose = False)
    fscore1 = clf_pos.fscore
    mio.Forest2Sql(clf_pos, X_p,2,1,advantage,fscore1,db)
    
    cm,clf_neg = mmll.clf_cross_validation\
    (eclf_neg,X_n,y_n,test_size=20,n_folds=20,train_test_labels = ttlabels,verbose = False)
    fscore2 = clf_neg.fscore
    mio.Forest2Sql(clf_neg,X_n,2,-1,advantage,fscore2,db)
    
    print "advantage-", advantage," f1-", fscore1, " f2-",fscore2
    
    return X_p, X_n,clf_pos,clf_neg

# <codecell>

def Forest2SqlClfPerc(eclf_pos,eclf_neg,advantage,db,df):  
    X_p = df[df.a14>0].copy()
    X_p.index = range(X_p.shape[0])
    X_p=X_p[['Ask','AskD','Near','Far','Spread',
     'D5', 'D555', 'D66', 'V1','V1n', 'V11', 'V11n',
     'V8','V8n','V8nn', 'a1','a4','a5']]

    X_n = df[df.a14<0].copy()
    X_n.index = range(X_n.shape[0])
    X_n=X_n[['Bid','BidD','Near','Far','Spread',
     'D4',  'D444', 'D66', 'V1','V1n', 'V11',  'V11n',
     'V8','V8n','V8nn', 'a1','a4','a5']]

    y_p = df.imbInd.copy()
    y_p[:] = 0
    if advantage>0:
        y_p[df.OPC_P/df.Ask_P-1.0>advantage/10000]=1
    else:
        y_p[df.OPC_P/df.Ask_P-1.0<advantage/10000]=1
    #y_p[df.OPC_P/df.Ref_P-1.0>advantage/10000]=1
    y_p = y_p[df.a14>0]
    y_p.index = range(y_p.shape[0])

    y_n = df.imbInd.copy()
    y_n[:] = 0
    if advantage>0:
        y_n[df.OPC_P/df.Bid_P-1<-advantage/10000]=1
    else:
        y_n[df.OPC_P/df.Bid_P-1>-advantage/10000]=1
    #y_n[df.OPC_P/df.Ref_P-1<-advantage/10000]=1
    y_n = y_n[df.a14<0]
    y_n.index = range(y_n.shape[0])
    
    ttlabesl_pos = df[df.a14>0].Date.copy();ttlabesl_pos.index = range(ttlabesl_pos.shape[0])
    ttlabesl_neg = df[df.a14<0].Date.copy();ttlabesl_neg.index = range(ttlabesl_neg.shape[0])
    
    cm,clf = mmll.clf_cross_validation\
    (eclf_pos,X_p,y_p,test_size=20,n_folds=20,train_test_labels = ttlabesl_pos,verbose = False)
    fscore1 = clf.fscore
    mio.Forest2Sql(clf, X_p,0,1,advantage,fscore1,db)
    
    cm,clf = mmll.clf_cross_validation\
    (eclf_neg,X_n,y_n,test_size=20,n_folds=20,train_test_labels = ttlabesl_neg,verbose = False)
    fscore2 = clf.fscore
    mio.Forest2Sql(clf,X_n,0,-1,advantage,fscore2,db)
    
    print "advantage-", advantage," f1-", fscore1, " f2-",fscore2

# <codecell>

def Forest2SqlClf(eclf_pos,eclf_neg,advantage,db,df):  
    X_p = df[df.a14>0].copy()
    X_p.index = range(X_p.shape[0])
    X_p=X_p[['Ask','AskD','Near','Far','Spread',
     'D5', 'D555', 'D66', 'V1','V1n', 'V11', 'V11n',
     'V8','V8n','V8nn', 'a1','a4','a5']]

    X_n = df[df.a14<0].copy()
    X_n.index = range(X_n.shape[0])
    X_n=X_n[['Bid','BidD','Near','Far','Spread',
     'D4',  'D444', 'D66', 'V1','V1n', 'V11',  'V11n',
     'V8','V8n','V8nn', 'a1','a4','a5']]
    
    y_p = df.imbInd.copy()
    y_p[:] = 0
    y_p[df.OPC_P-df.Ask_P>advantage/100]=1#y_p[df.OPC_P-df.Ref_P>advantage/100]=1#
    y_p = y_p[df.a14>0]
    y_p.index = range(y_p.shape[0])

    y_n = df.imbInd.copy()
    y_n[:] = 0
    y_n[df.OPC_P-df.Bid_P<-advantage/100]=1#y_n[df.OPC_P-df.Ref_P<-advantage/100]=1#
    y_n = y_n[df.a14<0]
    y_n.index = range(y_n.shape[0])
    
    ttlabesl_pos = df[df.a14>0].Date.copy();ttlabesl_pos.index = range(ttlabesl_pos.shape[0])
    ttlabesl_neg = df[df.a14<0].Date.copy();ttlabesl_neg.index = range(ttlabesl_neg.shape[0])
    
    cm,clf = mmll.clf_cross_validation\
    (eclf_pos,X_p,y_p,test_size=20,n_folds=20,train_test_labels = ttlabesl_pos,verbose = False)
    fscore1 = clf.fscore
    mio.Forest2Sql(clf, X_p,0,1,advantage,fscore1,db)
    
    cm,clf = mmll.clf_cross_validation\
    (eclf_neg,X_n,y_n,test_size=20,n_folds=20,train_test_labels = ttlabesl_neg,verbose = False)
    fscore2 = clf.fscore
    mio.Forest2Sql(clf,X_n,0,-1,advantage,fscore2,db)
    
    print "advantage-", advantage," f1-", fscore1, " f2-",fscore2

# <codecell>

db = '/home/user1/Desktop/Share2Windows/Qimb/TMP.sql'

# <codecell>

reload(mio)
mio.DropDB(db)

# <codecell>

reload(mio)
X_pos, X_neg,clf_pos,clf_neg = Forest2SqlZeroImbPerc(eclf_pos,eclf_neg,1.0,db,fdf)

mio.TestData2Sql(1, X_pos,ypln_pos,db)
mio.TestData2Sql(-1, X_neg,ypln_neg,db)

# <codecell>

reload(mio)
for advantage in reversed([30+x*5.0 for x in xrange(15)]):
    Forest2SqlClfPerc(eclf_pos,eclf_neg,-float(advantage),db,fdf)
for advantage in reversed([10+x*2.0 for x in xrange(10)]):
    Forest2SqlClfPerc(eclf_pos,eclf_neg,-float(advantage),db,fdf)
for advantage in reversed(range(10)):
    Forest2SqlClfPerc(eclf_pos,eclf_neg,-float(advantage),db,fdf)

# <codecell>

reload(mio)
for advantage in range(10):
    Forest2SqlClfPerc(eclf_pos,eclf_neg,float(advantage),db,fdf)
for advantage in [10+x*2.0 for x in xrange(10)]:
    Forest2SqlClfPerc(eclf_pos,eclf_neg,float(advantage),db,fdf)
for advantage in [30+x*5.0 for x in xrange(15)]:
    Forest2SqlClfPerc(eclf_pos,eclf_neg,float(advantage),db,fdf)

# <codecell>

reload(mio)
mio.TestData2Sql(1, X_pos,ypln_pos,db)
mio.TestData2Sql(-1, X_neg,ypln_neg,db)

# <codecell>


