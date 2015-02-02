# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import mmll
import EnsembleClassifier as ec
import mio

import numpy as np
import pandas as pd
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
f = '/home/user1/Desktop/Share2Windows/Qimb/rollBackFeatures.csv' 
                   
data = pd.read_csv(f, low_memory=False)

#imbalanceMsg = imbalanceMsg[imbalanceMsg.ImbInd!=0]
#imbalanceMsg.index = range(imbalanceMsg.shape[0])

print data.shape

data.head()

# <codecell>

list(data.columns)

# <codecell>

#Creating features
reload(qimbs)
fdf = qimbs.create_features44(data)
fdf.head()

# <codecell>

list(fdf.columns)

# <codecell>

X=fdf[[
'OPC_P',
'LastRef_P',
'LastNear_P',
'p1',
 'p2',
 'p3',
 'p4',
 'p5',
 'p6',
'p7',
 'p8',
 'p9',
 'p10',
    'p11',
 'p12',
 'p13',
 'p14',
 'p15',
 'p16',
 'p17',
 'p18',
'OPC_S', 'PrevCLC_S', 'LastImbSharesTraded', 'AllSharesTraded','MaxPairedS','v1',
 'v2',
 'v3',
 'v4',
 'v5',
 'v6']]

# <codecell>

X.describe()

# <codecell>

ret = 0.1
bidUp=pd.DataFrame(np.asarray(fdf.OPC_S),columns=['val']).astype(int)
bidUp.val = int(0) 
bidUp.val[fdf.Bid_Max-fdf.OPC_Bid_P>ret] =  int(1) 

bidDown=pd.DataFrame(np.asarray(fdf.OPC_S),columns=['val']).astype(int)
bidDown.val =  int(0) 
bidDown.val[fdf.OPC_Bid_P-fdf.Bid_Min>ret] =  int(1) 

askUp=pd.DataFrame(np.asarray(fdf.OPC_S),columns=['val']).astype(int)
askUp.val = int(0) 
askUp.val[fdf.Ask_Max-fdf.OPC_Ask_P>ret] =  int(1) 

askDown=pd.DataFrame(np.asarray(fdf.OPC_S),columns=['val']).astype(int)
askDown.val =  int(0) 
askDown.val[fdf.OPC_Ask_P-fdf.Ask_Min>ret] =  int(1) 

# <codecell>

ggplot(bidDown, aes('val')) + geom_histogram(binwidth=0.1,alpha=0.5, fill = 'green')+\
geom_histogram(askDown, aes('val'),binwidth=0.1,alpha=0.5, fill = 'red') 

# <codecell>

eclfUp = ec.EnsembleClassifier(
clfs=[
RF(min_samples_split = int(len(bidUp)*0.03),criterion='entropy',n_jobs=4)\
#,GBC(min_samples_split = len(bidUp)*0.03,init='zero')
#,LR(class_weight='auto',C=0.1)
])

eclfDown =ec.EnsembleClassifier(
clfs=[
RF(min_samples_split = int(len(bidDown)*0.03),criterion='entropy',n_jobs=4)\
#,GBC(min_samples_split = len(bidDown)*0.03,init='zero')
#,LR(class_weight='auto',C=0.1)
])

# <codecell>

reload(qimbs)
reload(mmll)
reload(ec)

# <codecell>

cm,clf = mmll.clf_cross_validation(eclfUp,X,bidUp.val,test_size=5,n_folds=10,train_test_labels = fdf.Date,verbose = True)

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

cm,clf = mmll.clf_cross_validation(eclfDown,X,bidDown.val,test_size=5,n_folds=10,train_test_labels = fdf.Date,verbose = True)

# <codecell>

cm,clf = mmll.clf_cross_validation(eclfUp,X,askUp.val,test_size=5,n_folds=10,train_test_labels = fdf.Date,verbose = True)

# <codecell>

cm,clf = mmll.clf_cross_validation(eclfDown,X,askDown.val,test_size=5,n_folds=10,train_test_labels = fdf.Date,verbose = True)

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






#Save configuration for C++

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

db = '/home/user1/Desktop/Share2Windows/RandomForestDatabasePAPosNeg.sql'
reload(mio)
mio.DropDB(db)

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


