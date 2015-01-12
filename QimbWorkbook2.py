# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import numpy as np
import pandas as pd
from ggplot import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier as RF

# <codecell>

reload(qimbs)

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

(n,b,p) = matplotlib.pyplot.hist(list(fdf.priceRange),bins=100,range=[0,50])

# <codecell>

fdf = fdf[fdf.priceRange>12]
fdf.index = range(fdf.shape[0])

# <codecell>

X_pos = fdf[fdf.a14>0]
X_pos.index = range(X_pos.shape[0])
X_pos=X_pos[['Ask','AskD','Near','Far','Spread',
 'D5', 'D555', 'D66', 'V1','V1n', 'V11', 'V11n',
 'V8','V8n','V8nn', 'a1','a4','a5','priceRange']]

X_neg = fdf[fdf.a14<0]
X_neg.index = range(X_neg.shape[0])
X_neg=X_neg[['Bid','BidD','Near','Far','Spread',
 'D4',  'D444', 'D66', 'V1','V1n', 'V11',  'V11n',
 'V8','V8n','V8nn', 'a1','a4','a5','priceRange']]

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
datesDF_pos = qimbs.dates_tmp_df(fdf[fdf.a14>0])
datesDF_neg = qimbs.dates_tmp_df(fdf[fdf.a14<0])

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

# <codecell>

ret = 0.2

ym_pos = fdf.OPC_P.copy()
ym_pos[:] = 0
ym_pos[fdf.OPC_P-fdf.Ask_P>ret] = 1
ym_pos = ym_pos[fdf.a14>0]
ym_pos.index = range(ym_pos.shape[0])

ym_neg = fdf.OPC_P.copy()
ym_neg[:] = 0
ym_neg[fdf.OPC_P-fdf.Bid_P<-ret] = 1
ym_neg = ym_neg[fdf.a14<0]
ym_neg.index = range(ym_neg.shape[0])

# <codecell>

ret = 0.2/100

yr_pos = fdf.OPC_P.copy()
yr_pos[:] = 0.0
yr_pos[fdf.OPC_P/fdf.Ask_P-1.0>ret]=1
yr_pos = yr_pos[fdf.a14>0]
yr_pos.index = range(yr_pos.shape[0])

yr_neg = fdf.OPC_P.copy()
yr_neg[:] = 0
yr_neg[fdf.OPC_P/fdf.Bid_P-1<-ret]=1
yr_neg = yr_neg[fdf.a14<0]
yr_neg.index = range(yr_neg.shape[0])

# <codecell>

(n,b,p) = matplotlib.pyplot.hist(list(ym_pos),bins=10,range=[0,1])

# <codecell>

X_pos.head()

# <codecell>

qimbs.OneModelResults('NN', X_pos,y_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults('NN', X_neg,y_neg,ERRORS,dates,datesDF_neg)

# <codecell>

qimbs.OneModelResults('COMB', X_pos,ym_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults('COMB', X_neg,ym_neg,ERRORS,dates,datesDF_neg)

# <codecell>

reload(qimbs)
qimbs.OneModelResults(RF, X_pos,ym_pos,ERRORS,dates,datesDF_pos)

# <codecell>

reload(qimbs)
qimbs.OneModelResults(RF, X_neg,ym_neg,ERRORS,dates,datesDF_neg)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier as GBC
qimbs.OneModelResults(GBC, X_pos,yr_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults(GBC, X_neg,yr_neg,ERRORS,dates,datesDF_neg)

# <codecell>

qimbs.OneModelResults('COMB', X_pos,ym_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults('COMB', X_neg,ym_neg,ERRORS,dates,datesDF_neg)

# <codecell>






#Save configuration for C++

# <codecell>

db = '/home/user1/Desktop/Share2Windows/RandomForestDatabase.sql'
reload(qimbs)
qimbs.DropDB(db)

# <codecell>

reload(qimbs)
qimbs.TestData2Sql(1, X_pos,ypln_pos,db)
qimbs.TestData2Sql(-1, X_neg,ypln_neg,db)

# <codecell>

reload(qimbs)
Forest2SqlClf(0.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(1.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(2.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(3.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(4.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(5.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(7.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(9.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(11.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(13.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(15.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(17.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)
Forest2SqlClf(19.0,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg)

# <codecell>

reload(qimbs)
Forest2SqlReg(db,fdf,X_pos,X_neg)

# <codecell>

def Forest2SqlClf(advantage,db,df,priceRangeMin,priceRangeMax):
    df = df[(df.priceRange>=priceRangeMin) & (df.priceRange<priceRangeMax)]
    df.index = range(df.shape[0])
    
    X_p = df[df.a14>0]
    X_p.index = range(X_p.shape[0])
    X_p=X_p[['Ask','AskD','Near','Far','Spread',
     'D5', 'D555', 'D66', 'V1','V1n', 'V11', 'V11n',
     'V8','V8n','V8nn', 'a1','a4','a5']]

    X_n = df[df.a14<0]
    X_n.index = range(X_n.shape[0])
    X_n=X_n[['Bid','BidD','Near','Far','Spread',
     'D4',  'D444', 'D66', 'V1','V1n', 'V11',  'V11n',
     'V8','V8n','V8nn', 'a1','a4','a5']]

    y_p = df.OPC_P.copy()
    y_p[:] = 0
    y_p[df.OPC_P>df.Ask_P+advantage/100] = 1
    y_p = y_pos[df.a14>0]
    y_p.index = range(y_p.shape[0])

    y_n = df.OPC_P.copy()
    y_n[:] = 0
    y_n[df.OPC_P<df.Bid_P-advantage/100] = 1
    y_n = y_n[df.a14<0]
    y_n.index = range(y_n.shape[0])
    
    dates = sorted(list(set(df.Date)))
    datesDF_p = qimbs.dates_tmp_df(df[df.a14>0])
    datesDF_n = qimbs.dates_tmp_df(df[df.a14<0])
    
    trainError, testError, cm, clf, fscore1 = \
    qimbs.run_cv_proba(X_p,y_p,RF,20,20,dates,datesDF_p)
    qimbs.Forest2Sql(clf, X_p,0,1,advantage,fscore1,db)
    
    trainError, testError, cm, clf, fscore2 = \
    qimbs.run_cv_proba(X_n,y_n,RF,20,20,dates,datesDF_n)
    qimbs.Forest2Sql(clf,X_n,0,-1,advantage,fscore2,db)
    
    print "f1-", fscore1, " f2-",fscore2

# <codecell>

def Forest2SqlReg(db,fdf,X_pos,X_neg):
    #Random forest for regression
    
    from sklearn.ensemble import RandomForestRegressor as RFR
    '''Xpp = X_pos.copy()
    Xpp = Xpp[ypln_pos>0]
    Xpp.index = range(Xpp.shape[0])

    clf = RFR(min_samples_split = ypln_pos[ypln_pos>0].shape[0]*0.05)
    clf.fit(Xpp, ypln_pos[ypln_pos>0])
    qimbs.Forest2Txt(clf, Xpp.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/PP')

    clf = GBR(min_samples_split = ypln_pos[ypln_pos>0].shape[0]*0.05, loss='huber',init='zero',learning_rate=0.1)
    clf.fit(Xpp, ypln_pos[ypln_pos>0])
    qimbs.Forest2Txt(clf, Xpp.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/PP')'''

    #------------------------------------------
    Xpn = X_pos.copy()
    Xpn = Xpn[ypln_pos<0]
    Xpn.index = range(Xpn.shape[0])
    
    clf, r2 = run_reg(X_pos,ypln_pos,RFR,20,20,dates,datesDF_pos)
    qimbs.Forest2Txt(clf, Xpn.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/PN')
    qimbs.Forest2Sql(clf, Xpn.ix[:,:],1,1,0,r2,db)

    clf, r2 = run_reg(X_pos,ypln_pos,GBR,20,20,dates,datesDF_pos)
    qimbs.Forest2Txt(clf, Xpn.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/PN')
    qimbs.Forest2Sql(clf, Xpn.ix[:,:],1,1,0,r2,db)

    #------------------------------------------
    '''Xnp = X_neg.copy()
    Xnp = Xnp[ypln_neg>0]
    Xnp.index = range(Xnp.shape[0])

    clf = RFR(min_samples_split = ypln_neg[ypln_neg>0].shape[0]*0.05)
    clf.fit(Xnp, ypln_neg[ypln_neg>0])
    qimbs.Forest2Txt(clf, Xnp.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/NP')

    clf = GBR(min_samples_split = ypln_neg[ypln_neg>0].shape[0]*0.05, loss='huber',init='zero',learning_rate=0.1)
    clf.fit(Xnp, ypln_neg[ypln_neg>0])
    qimbs.Forest2Txt(clf, Xnp.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/NP')'''

    #------------------------------------------
    Xnn = X_neg.copy()
    Xnn = Xnn[ypln_neg<0]
    Xnn.index = range(Xnn.shape[0])

    clf, r2 = run_reg(X_neg,ypln_neg,RFR,20,20,dates,datesDF_neg)
    qimbs.Forest2Txt(clf, Xnn.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/NN')
    qimbs.Forest2Sql(clf, Xnn.ix[:,:],1,-1,0,r2,db)

    clf, r2 = run_reg(X_neg,ypln_neg,GBR,20,20,dates,datesDF_neg)
    qimbs.Forest2Txt(clf, Xnn.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/NN')
    qimbs.Forest2Sql(clf, Xnn.ix[:,:],1,-1,0,r2,db)

