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
f = '/home/user1/PyProjects/imbalanceMsg.csv' 
                   
imbalanceMsg = pd.read_csv(f, low_memory=False)

#imbalanceMsg = imbalanceMsg[imbalanceMsg.ImbInd!=0]
#imbalanceMsg.index = range(imbalanceMsg.shape[0])

print imbalanceMsg.shape

imbalanceMsg.head()

# <codecell>

sum(map(int,imbalanceMsg.ImbInd==0))

# <codecell>

#Creating features
reload(qimbs)
fdf = qimbs.create_features33(imbalanceMsg)
fdf.head()

# <codecell>

fdf.priceRange[fdf.priceRange>1]=1

# <codecell>

(n,b,p) = matplotlib.pyplot.hist(list(fdf.priceRange),bins=100,range=[0,50])

# <codecell>

#fdf = fdf[fdf.priceRange==0]
#fdf.index = range(fdf.shape[0])

# <codecell>

X_pos = fdf[fdf.a14>0]
X_pos.index = range(X_pos.shape[0])
X_pos=X_pos[['Ask','AskD','Near','Far','Spread',
 'D5', 'D555', 'D66', 'V1','V1n', 'V11', 'V11n',
 'V8','V8n','V8nn', 'a1','a4','a5']]

X_neg = fdf[fdf.a14<0]
X_neg.index = range(X_neg.shape[0])
X_neg=X_neg[['Bid','BidD','Near','Far','Spread',
 'D4',  'D444', 'D66', 'V1','V1n', 'V11',  'V11n',
 'V8','V8n','V8nn', 'a1','a4','a5']]

y_pos = fdf.OPC_P>fdf.Ask_P
y_pos = y_pos[fdf.a14>0]
y_pos.index = range(y_pos.shape[0])

y_neg = fdf.OPC_P<fdf.Bid_P
y_neg = y_neg[fdf.a14<0]
y_neg.index = range(y_neg.shape[0])

ypln_pos = fdf.OPC_P/fdf.Ask_P-1
ypln_pos = ypln_pos[fdf.a14>0]
ypln_pos.index = range(ypln_pos.shape[0])

ypln_neg = 1-fdf.OPC_P/fdf.Bid_P
ypln_neg = ypln_neg[fdf.a14<0]
ypln_neg.index = range(ypln_neg.shape[0])

dates = sorted(list(set(fdf.Date)))

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

# <codecell>

ret = 0.01

ym_pos = fdf.imbInd.copy()
ym_pos[:] = 0
ym_pos[fdf.OPC_P-fdf.Ask_P>ret] = 1
ym_pos = ym_pos[fdf.a14>0]
ym_pos.index = range(ym_pos.shape[0])

ym_neg = fdf.imbInd.copy()
ym_neg[:] = 0
ym_neg[fdf.OPC_P-fdf.Bid_P<-ret] = 1
ym_neg = ym_neg[fdf.a14<0]
ym_neg.index = range(ym_neg.shape[0])

# <codecell>

ret = 0.01/100

yr_pos = fdf.imbInd.copy()
yr_pos[:] = 0
yr_pos[fdf.OPC_P/fdf.Ask_P-1.0>ret]=1
yr_pos = yr_pos[fdf.a14>0]
yr_pos.index = range(yr_pos.shape[0])

yr_neg = fdf.imbInd.copy()
yr_neg[:] = 0
yr_neg[fdf.OPC_P/fdf.Bid_P-1<-ret]=1
yr_neg = yr_neg[fdf.a14<0]
yr_neg.index = range(yr_neg.shape[0])

# <codecell>

(n,b,p) = matplotlib.pyplot.hist(list(ym_pos),bins=10,range=[0,1])

# <codecell>

X_pos.head()

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

labels_pos = fdf[fdf.a14>0].priceRange; labels_pos.index = range(labels_pos.shape[0])
labels_neg = fdf[fdf.a14<0].priceRange; labels_neg.index = range(labels_neg.shape[0])
ttlabesl_pos = fdf[fdf.a14>0].Date;ttlabesl_pos.index = range(ttlabesl_pos.shape[0])
ttlabesl_neg = fdf[fdf.a14<0].Date;ttlabesl_neg.index = range(ttlabesl_neg.shape[0])

# <codecell>

reload(qimbs)
reload(mmll)
reload(ec)

# <codecell>

cm,clf = mmll.clf_cross_validation(eclf_pos,X_pos,yr_pos,test_size=5,n_folds=50,train_test_labels = ttlabesl_pos,verbose = True)

# <codecell>

cm,clf = mmll.clf_cross_validation(eclf_neg,X_neg,yr_neg,test_size=5,n_folds=50,train_test_labels = ttlabesl_neg,verbose = True)

# <codecell>







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
    y_p[df.OPC_P/df.Ask_P-1.0>advantage/10000]=1
    y_p = y_p[df.a14>0]
    y_p.index = range(y_p.shape[0])

    y_n = df.imbInd.copy()
    y_n[:] = 0
    y_n[df.OPC_P/df.Bid_P-1<-advantage/10000]=1
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
    
    print "f1-", fscore1, " f2-",fscore2

# <codecell>

db = '/home/user1/Desktop/Share2Windows/RandomForestDatabasePercentageAdvantage.sql'
reload(mio)
mio.DropDB(db)

# <codecell>

reload(mio)
Forest2SqlClfPerc(eclf_pos,eclf_neg,0.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,1.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,2.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,3.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,4.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,5.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,7.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,9.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,11.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,13.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,15.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,17.0,db,fdf)
Forest2SqlClfPerc(eclf_pos,eclf_neg,19.0,db,fdf)

# <codecell>

reload(mio)
mio.TestData2Sql(1, X_pos,ypln_pos,db)
mio.TestData2Sql(-1, X_neg,ypln_neg,db)

# <codecell>


