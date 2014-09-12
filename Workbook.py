# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import numpy as np
import pandas as pd
from ggplot import *

# <codecell>

reload(qimbs)

# <codecell>

#!Importing data
df = pd.DataFrame()
for i in range(5,7):
    df = df.append(qimbs.import_month(i))
    print i
print df.shape
df.index = range(df.shape[0])

#Adding Timestamp column
df = qimbs.create_timestamp(df)

# <codecell>

#Getting imbalance info
imbalanceMsg = qimbs.get_imbelanceMSG2(df,0)
imbalanceMsg = imbalanceMsg[
    (imbalanceMsg.Ask_P - imbalanceMsg.Bid_P < 0.2 * 
    (imbalanceMsg.Ask_P + imbalanceMsg.Bid_P)*0.5) ]
imbalanceMsg.index = range(imbalanceMsg.shape[0])

# <codecell>

#Creating features
fdf,Features = qimbs.create_features(imbalanceMsg)
X = fdf[[ 'Spread', 'D1', 'D2', 'D3', 'D4', 'D5','D44', 'D55','D444', 'D555', 'D6', 'D66','D7', 'V1', 'V11',
         'V2', 'V3', 'V4', 'V5', 'V6', 'V7','V8','V9', 'a1', 'a2', 'a3',
 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13','a14']]
X['Bias'] = np.ones((X.shape[0],1))

y = fdf['Move']

dates = sorted(list(set(fdf.Date)))
datesDF = qimbs.dates_tmp_df(fdf)

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

# <codecell>

#Creating features
fdf,Features = qimbs.create_features3(imbalanceMsg)

X = fdf[['Bid','Ask','Near','Far','PrevCLC','Spread',
 'D3', 'D4', 'D5', 'D444', 'D555', 'D7', 'D66', 'V1', 'V11', 
 'V8', 'a3', 'a4', 'a14' ]]

y = fdf['Move']

dates = sorted(list(set(fdf.Date)))
datesDF = qimbs.dates_tmp_df(fdf)

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

# <codecell>

ggplot(fdf,aes(x='Move')) + geom_histogram(binwidth = 0.05) 

# <codecell>

#Benchmark - sell when imb<0, buy when imb>0 
qimbs.OneModelResults('B', X,y,ERRORS,dates,datesDF)

# <codecell>

#Apply Random Forest
from sklearn.ensemble import RandomForestClassifier as RF
print "Random forest:"
qimbs.OneModelResults(RF, X,y,ERRORS,dates,datesDF)

# <codecell>

from sklearn import feature_selection
from sklearn.svm import SVC 
fs = feature_selection.RFE(estimator=SVC(kernel='linear'),n_features_to_select=5)
fs.fit(X,y)
print list(X.columns[fs.ranking_==1])

# <codecell>

#Lets see features importance and try to reduce the number of feature for RF algo
from sklearn.ensemble import RandomForestClassifier as RF
clf = RF(min_samples_split=0.1*X.shape[0])
clf.fit(X,y)

fi = pd.DataFrame()
fi['Feature'] = list(X.columns)
fi['Impotrance'] = clf.feature_importances_
fi=fi.sort(columns=['Impotrance'],ascending=False)
fi['Index'] = range(X.shape[1])

ggplot(fi,aes('Index','Impotrance',label='Feature')) +\
geom_point() + geom_text(vjust=0.005)

# <codecell>

print 'Main features are:'
for f in list(fi['Feature'][:12]):
    print '%s  %s' %(f,Features[f])

# <codecell>

#Get error for these features bags
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix

error1 = []
error2 = []
error3 = []
nfeatures = range(1, X.shape[1])
labels = list(set(y))
for nfeature in nfeatures:
    print nfeature
    X_r = X[fi['Feature'][:nfeature]]
    
    err1 = []
    err2 = []
    err3 = []
    for i in range(2):  
        r = range(len(dates))
        np.random.shuffle(r)
        test_days = r[:1] 
        train_days = r[1:] 
               
        Xtrain = X_r.ix[datesDF.ix[train_days],:]
        Xtest = X_r.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]]
        
        reg1 = RF()
        reg1.fit(Xtrain, ytrain)
        test_pr1 = qimbs.Precision_Recall(confusion_matrix(ytest,reg1.predict(Xtest),labels).astype(float))
        err1.append(1-test_pr1[2])  
        
        reg2 = LR(class_weight='auto')
        reg2.fit(Xtrain, ytrain)
        test_pr2 = qimbs.Precision_Recall(confusion_matrix(ytest,reg2.predict(Xtest),labels).astype(float))
        err2.append(1-test_pr2[2]) 
        
        reg3 = SVC(class_weight='auto')
        reg3.fit(Xtrain, ytrain)
        test_pr3 = qimbs.Precision_Recall(confusion_matrix(ytest,reg3.predict(Xtest),labels).astype(float))
        err3.append(1-test_pr3[2])  
        
    error1.append(np.mean(err1))
    error2.append(np.mean(err2))
    error3.append(np.mean(err3))
    

# <codecell>

p_df=pd.DataFrame()
p_df['I']=range(1, X.shape[1])
p_df['RF']=error1
p_df['LR']=error2
p_df['SVC']=error3
p_df_m = pd.melt(p_df, id_vars=['I'])

ggplot(p_df_m,aes(x='I', y='value', colour='variable')) + geom_line() + geom_point() +\
xlab('n important features selected')+ylab('Error')+ggtitle('Features importance')

# <codecell>

#Lets run RF with les features
qimbs.OneModelResults(RF, X[fi['Feature'][:12]],y, ERRORS,dates,datesDF)

# <codecell>

#Apply Logistic Regression
from sklearn.linear_model import LogisticRegression as LR
print "Logistic Regression:"
reload(qimbs)
qimbs.OneModelResults(LR, X,y,ERRORS,dates,datesDF, n_ensembles=5, test_size_ensemble=0.2)

# <codecell>

from sklearn.svm import SVC
print "SVC:"
qimbs.OneModelResults(SVC, X[fi['Feature'][:10]], y,ERRORS,dates,datesDF, n_ensembles=5, test_size_ensemble=0.2)

# <codecell>

ERRORS

# <codecell>

#Models comparison
ERRORS=ERRORS.sort(['TrainError', 'TestError'])
#ERRORS=ERRORS[:-1]
ggplot(ERRORS, aes('TrainError', 'TestError')) + geom_point() + \
ggtitle('Model comparison') + xlab("TrainError") +\
ylab("TestError") +geom_text(aes(label='Model'),hjust=0, vjust=0)\
+ ylim(0,0.5)+ xlim(0,0.5) + geom_abline(intercept = 0.217, slope = 0\
                                         ,color='red')

# <codecell>

#Lets compare models in money
reload(qimbs)
#Benchmark
Signals =  qimbs.get_signals1(imbalanceMsg,X,y,'B',dates,datesDF)
Symbols = sorted(list(set(df.Symbol)))
SymbolsInd=dict()
for i in range(len(Symbols)):
    SymbolsInd[Symbols[i]]=i
    
T = zeros((len(Symbols),len(Symbols)))
TN = zeros((len(Symbols),len(Symbols)))
result0 = qimbs.get_performance(Signals,df,dates,SymbolsInd,T,TN,0)
ggplot(result0, aes('Date','Pnl')) + geom_point() + ggtitle('Sum=%s' % result0.Pnl.sum()) + geom_line()

# <codecell>

#LR
from sklearn.linear_model import LogisticRegression as LR
Signals =  qimbs.get_signals1(imbalanceMsg,X,y,LR,dates,datesDF)
Symbols = sorted(list(set(df.Symbol)))
SymbolsInd=dict()
for i in range(len(Symbols)):
    SymbolsInd[Symbols[i]]=i
T = zeros((len(Symbols),len(Symbols)))
TN = zeros((len(Symbols),len(Symbols)))
result1 = qimbs.get_performance(Signals,df,dates,SymbolsInd,T,TN,0)
ggplot(result1, aes('Date','Pnl')) + geom_point() + ggtitle('Sum=%s' % result1.Pnl.sum()) + geom_line()

# <codecell>

#RandomForest
Signals =  qimbs.get_signals1(imbalanceMsg,X[fi['Feature'][:10]],y,RF,dates,datesDF)
Symbols = sorted(list(set(df.Symbol)))
SymbolsInd=dict()
for i in range(len(Symbols)):
    SymbolsInd[Symbols[i]]=i
T = zeros((len(Symbols),len(Symbols)))
TN = zeros((len(Symbols),len(Symbols)))
result2 = qimbs.get_performance(Signals,df,dates,SymbolsInd,T,TN,0)
ggplot(result2, aes('Date','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>

#SVC
from sklearn.svm import SVC
Signals =  qimbs.get_signals1(imbalanceMsg, X[fi['Feature'][:10]],y,SVC,dates,datesDF)
Symbols = sorted(list(set(df.Symbol)))
SymbolsInd=dict()
for i in range(len(Symbols)):
    SymbolsInd[Symbols[i]]=i
T = zeros((len(Symbols),len(Symbols)))
TN = zeros((len(Symbols),len(Symbols)))
result1 = qimbs.get_performance(Signals,df,dates,SymbolsInd,T,TN,0)
ggplot(result1, aes('Date','Pnl')) + geom_point() + ggtitle('Sum=%s' % result1.Pnl.sum()) + geom_line()

# <codecell>

from IPython.core.display import Image 
Image(filename='/home/user/PyProjects/Results/1.png') 

# <codecell>

from sklearn.ensemble import RandomForestClassifier as RF
clf = RF(min_samples_split = X.shape[0]*0.05)
clf.fit(X[fi['Feature'][:10]],y)
qimbs.Forest2Txt(clf, X[fi['Feature'][:10]].ix[0:100,:],'/home/user1/Desktop/Share2Windows')

# <codecell>


