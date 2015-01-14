# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import numpy as np
import pandas as pd
from ggplot import *
from sklearn.preprocessing import PolynomialFeatures

# <codecell>

reload(qimbs)

# <codecell>

from IPython import parallel
clients = parallel.Client()
clients.block = True  # use synchronous computations
print clients.ids

# <codecell>

#data2 = clients[0].apply(qimbs.import_month,2)
#data3 = clients[1].apply(qimbs.import_month,3)
#data4 = clients[2].apply(qimbs.import_month,4)
#data5 = clients[3].apply(qimbs.import_month,5)

# <codecell>

#data6 = clients[0].apply(qimbs.import_month,6)

# <codecell>

#!Importing data
df = pd.DataFrame()
for i in range(7,8):
    df = df.append(qimbs.import_month(i))
    print i
    
#df = df.append(data2); del data2
#df = df.append(data3); del data3
#df = df.append(data4); del data4
#df = df.append(data5); del data5
#df = df.append(data6); del data6
print df.shape
df.index = range(df.shape[0])

# <codecell>

#Getting imbalance info
reload(qimbs)
imbalanceMsg = qimbs.get_imbalanceMSG2(df,0)
imbalanceMsg = imbalanceMsg[
    (imbalanceMsg.Ask_P - imbalanceMsg.Bid_P < 0.1 * (imbalanceMsg.Ask_P + imbalanceMsg.Bid_P)*0.5) ]
imbalanceMsg.index = range(imbalanceMsg.shape[0])
imbalanceMsg.shape

# <codecell>

imbalanceMsg_=imbalanceMsg.copy()

# <codecell>

imbalanceMsg_=imbalanceMsg_.append(imbalanceMsg)
imbalanceMsg=imbalanceMsg_
imbalanceMsg.index = range(imbalanceMsg.shape[0])
imbalanceMsg.shape

# <codecell>

#Procees c++ created features
f = '/home/user1/PyProjects/imbalanceMsg.csv' 
                   
imbalanceMsg = pd.read_csv(f, low_memory=False)

print imbalanceMsg.shape

imbalanceMsg.tail()

# <codecell>

#Creating features
reload(qimbs)
fdf = qimbs.create_features33(imbalanceMsg)

# <codecell>


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
datesDF_pos = qimbs.dates_tmp_df(fdf[fdf.a14>0])
datesDF_neg = qimbs.dates_tmp_df(fdf[fdf.a14<0])

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

# <codecell>

ym_pos = fdf.OPC_P.copy()
ym_pos[:] = 0
ym_pos[fdf.OPC_P>fdf.Ask_P+0.1] = 1
ym_pos = ym_pos[fdf.a14>0]
ym_pos.index = range(y_pos.shape[0])

ym_neg = fdf.OPC_P.copy()
ym_neg[:] = 0
ym_neg[fdf.OPC_P<fdf.Bid_P-0.1] = 1
ym_neg = ym_neg[fdf.a14<0]
ym_neg.index = range(y_neg.shape[0])

# <codecell>

X_pos.tail()

# <codecell>

ggplot(fdf,aes(x='p')) + geom_histogram() #binwidth = 0.05

# <codecell>

ggplot(fdf[fdf.y>0],aes(x='D4',y='D5')) + geom_point(alpha=0.4, color='green',size=2) + \
geom_point(fdf[fdf.y<0],aes(x='D4',y='D5'),alpha=0.4, color='red',size=2) + \
geom_point(fdf[fdf.y==0],aes(x='D4',y='D5'),alpha=0.4, color='blue',size=2) 

# <codecell>

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3)
kmeans.fit(fdf.Mid_P)

# <codecell>

y_pos = fdf.y>0
y_pos = y_pos[(fdf.a14>0) & (fdf.ImbInd!=0)]
y_pos.index = range(y_pos.shape[0])

X_pos = fdf[(fdf.a14>0) & (fdf.ImbInd!=0)]
X_pos.index = range(X_pos.shape[0])
X_pos=X_pos[['Ask','AskD','Near','Far','Spread',
 'D5', 'D555', 'D66', 'V1','V1n', 'V11', 'V11n',
 'V8','V8n','V8nn', 'a1','a4','a5']]

y_neg = fdf.y<0
y_neg = y_neg[(fdf.a14<0) & (fdf.ImbInd!=0)]
y_neg.index = range(y_neg.shape[0])

X_neg = fdf[(fdf.a14<0) & (fdf.ImbInd!=0)]
X_neg.index = range(X_neg.shape[0])
X_neg=X_neg[['Bid','BidD','Near','Far','Spread',
 'D4',  'D444', 'D66', 'V1','V1n', 'V11',  'V11n',
 'V8','V8n','V8nn', 'a1','a4','a5']]

dates = sorted(list(set(fdf.Date)))
datesDF_pos = qimbs.dates_tmp_df(fdf[(fdf.a14>0) & (fdf.ImbInd!=0)])
datesDF_neg = qimbs.dates_tmp_df(fdf[(fdf.a14<0) & (fdf.ImbInd!=0)])

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

# <codecell>

ym_pos = fdf.OpenP.copy()
ym_pos[:] = 0
ym_pos[fdf.OpenP>fdf.AskP+0.05] = 1
ym_pos = ym_pos[(fdf.a14>0) & (fdf.ImbInd==0)]
ym_pos.index = range(ym_pos.shape[0])

ym_neg = fdf.OpenP.copy()
ym_neg[:] = 0
ym_neg[fdf.OpenP<fdf.BidP-0.05] = 1
ym_neg = ym_neg[(fdf.a14<0) & (fdf.ImbInd==0)]
ym_neg.index = range(ym_neg.shape[0])

# <codecell>

#For automatic features generation
y_pos = fdf.y>0
y_pos = y_pos[fdf.ImbSign>0]
y_pos.index = range(y_pos.shape[0])

X_pos = fdf[fdf.ImbSign>0]
X_pos.index = range(X_pos.shape[0])
X_pos1=X_pos[['CloseP','Bid','Ask','Ref','Far','Near','ClosePSPY','BidSPY','AskSPY']]
X_pos2=X_pos[['BidS','AskS','ImbS','AImbS','PairedS']]
autoX_pos1 = autocreateFeatures(X_pos1,y_pos,1)
autoX_pos1.columns = ["a"+str(x) for x in range(autoX_pos1.shape[1])]
autoX_pos2 = autocreateFeatures(X_pos2,y_pos,1)
autoX_pos2.columns = ["b"+str(x) for x in range(autoX_pos2.shape[1])]
autoX_pos = pd.concat([autoX_pos1, autoX_pos2], axis=1)
#X_pos_Benchmark = createFeaturesBench(X_pos)


y_neg = fdf.y<0
y_neg = y_neg[fdf.ImbSign<0]
y_neg.index = range(y_neg.shape[0])

X_neg = fdf[fdf.ImbSign<0]
X_neg.index = range(X_neg.shape[0])
X_neg1=X_neg[['CloseP','Bid','Ask','Ref','Far','Near','ClosePSPY','BidSPY','AskSPY']]
X_neg2=X_neg[['BidS','AskS','ImbS','AImbS','PairedS']]
autoX_neg1 = autocreateFeatures(X_neg1,y_neg,1)
autoX_neg1.columns = ["a"+str(x) for x in range(autoX_neg1.shape[1])]
autoX_neg2 = autocreateFeatures(X_neg2,y_neg,1)
autoX_neg2.columns = ["b"+str(x) for x in range(autoX_neg2.shape[1])]
autoX_neg = pd.concat([autoX_neg1, autoX_neg2], axis=1)
#X_neg_Benchmark = createFeaturesBench(X_neg)


dates = sorted(list(set(fdf.Date)))
datesDF_pos = qimbs.dates_tmp_df(fdf[fdf.ImbSign>0])
datesDF_neg = qimbs.dates_tmp_df(fdf[fdf.ImbSign<0])

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

# <codecell>

qimbs.OneModelResults('NN', X_pos,y_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults('NN', X_neg,y_neg,ERRORS,dates,datesDF_neg)

# <codecell>

qimbs.OneModelResults('COMB', X_pos,y_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults('COMB', X_neg,ym_neg,ERRORS,dates,datesDF_neg)

# <codecell>

X_pos.describe()

# <codecell>

from sklearn.ensemble import RandomForestClassifier as RF
reload(qimbs)
qimbs.OneModelResults(RF, X_pos,ym_pos,ERRORS,dates,datesDF_pos)

# <codecell>

reload(qimbs)
qimbs.OneModelResults(RF, X_neg,ym_neg,ERRORS,dates,datesDF_neg)

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier as GBC
qimbs.OneModelResults(GBC, X_pos,ym_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults(GBC, X_neg,ym_neg,ERRORS,dates,datesDF_neg)

# <codecell>

qimbs.OneModelResults('COMB', X_pos,ym_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults('COMB', X_neg,ym_neg,ERRORS,dates,datesDF_neg)

# <codecell>

qimbs.OneModelResults(RF, X_pos,ym_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults(RF, X_neg,ym_neg,ERRORS,dates,datesDF_neg)

# <codecell>

ym_pos.hist()

# <codecell>

from sklearn.ensemble import GradientBoostingClassifier as GBC
qimbs.OneModelResults(GBC, X_pos,ym_pos,ERRORS,dates,datesDF_pos)

# <codecell>

qimbs.OneModelResults(GBC, X_neg,ym_neg,ERRORS,dates,datesDF_neg)

# <codecell>

#Lets see features importance and try to reduce the number of feature for RF algo
from sklearn.ensemble import RandomForestClassifier as RF
clf = RF(min_samples_split = X_pos.shape[0]*0.05, criterion = 'entropy',n_estimators =10)
#Stage 2
newX_pos=X_pos#createFeatures34(X_pos2)
clf.fit(newX_pos,y_pos)

fi = pd.DataFrame()
fi['Feature'] = list(newX_pos.columns)
fi['Impotrance'] = clf.feature_importances_
fi=fi.sort(columns=['Impotrance'],ascending=False)
fi['Index'] = range(newX_pos.shape[1])
fi.index = fi['Index']

'''for i in range(fi.shape[0]):
    if (fi['Impotrance'][i]<0.005):
        break
    #print fi['Feature'][i]

newX_pos = newX_pos[fi['Feature'][:i]]'''

#Stage 2
'''poly = PolynomialFeatures(2)
newX_pos_2=pd.DataFrame(poly.fit_transform(newX_pos))
clf.fit(newX_pos_2,y_pos)

fi = pd.DataFrame()
fi['Feature'] = list(newX_pos_2.columns)
fi['Impotrance'] = clf.feature_importances_
fi=fi.sort(columns=['Impotrance'],ascending=False)
fi['Index'] = range(newX_pos_2.shape[1])
fi.index = fi['Index']

for i in range(fi.shape[0]):
    if (fi['Impotrance'][i]<0.01):
        break
    #print fi['Feature'][i]

newX_pos_2 = newX_pos_2[fi['Feature'][:i]]'''
   
ggplot(fi,aes('Index','Impotrance',label='Feature')) +\
geom_point() + geom_text(vjust=0.005)

# <codecell>

clf = RF(min_samples_split = X_neg.shape[0]*0.05, criterion = 'entropy',n_estimators =10)
newX_neg=X_neg#createFeatures35(X_neg)
clf.fit(newX_neg,y_neg)

fi = pd.DataFrame()
fi['Feature'] = list(newX_neg.columns)
fi['Impotrance'] = clf.feature_importances_
fi=fi.sort(columns=['Impotrance'],ascending=False)
fi['Index'] = range(newX_neg.shape[1])
fi.index = fi['Index']

for i in range(fi.shape[0]):
    if (fi['Impotrance'][i]<0.005):
        break
    #print fi['Feature'][i]
    
newX_neg = newX_neg[fi['Feature'][:i]]

#Stage 2
poly = PolynomialFeatures(2)
newX_neg_2=pd.DataFrame(poly.fit_transform(newX_neg))
clf.fit(newX_neg_2,y_neg)

fi = pd.DataFrame()
fi['Feature'] = list(newX_neg_2.columns)
fi['Impotrance'] = clf.feature_importances_
fi=fi.sort(columns=['Impotrance'],ascending=False)
fi['Index'] = range(newX_neg_2.shape[1])
fi.index = fi['Index']

for i in range(fi.shape[0]):
    if (fi['Impotrance'][i]<0.01):
        break
    #print fi['Feature'][i]

newX_neg_2 = newX_neg_2[fi['Feature'][:i]]
    
ggplot(fi[fi['Impotrance']>0.01],aes('Index','Impotrance',label='Feature')) +\
geom_point() + geom_text(vjust=0.005)

# <codecell>














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
qimbs.OneModelResults(RF, X[fi['Feature'][:15]],np.abs(y), ERRORS,dates,datesDF)

# <codecell>

#Apply Logistic Regression
from sklearn.linear_model import LogisticRegression as LR
print "Logistic Regression:"
reload(qimbs)
qimbs.OneModelResults(LR, X_pos,y_pos,ERRORS,dates,datesDF_pos)

# <codecell>

from sklearn.svm import SVC
print "SVC:"
qimbs.OneModelResults(SVC, X_pos, ym_pos,ERRORS,dates,datesDF_pos)

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

reload(qimbs)

















# <codecell>

#Lets compare models in money
reload(qimbs)
#Benchmark
Signals =  qimbs.get_signals1(imbalanceMsg,X,y,'B',dates,datesDF)
result0 = qimbs.get_performance(Signals,df,dates,0)
result0['I'] = result0.index
ggplot(result0, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result0.Pnl.sum()) + geom_line()

# <codecell>

#RandomForest
Signals =  qimbs.get_signals1(imbalanceMsg,X,y,RF,dates,datesDF)
result2 = qimbs.get_performance(Signals,df,dates,0)
result2['I'] = result2.index
ggplot(result2, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>

#RandomForest
Signals =  qimbs.get_signals1(imbalanceMsg,X,np.abs(y),RF,dates,datesDF)
result2 = qimbs.get_performance(Signals,df,dates,0)
result2['I'] = result2.index
ggplot(result2, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>

reload(qimbs)

# <codecell>

#RandomForest CMove
Signals =  qimbs.get_signals_proba(imbalanceMsg,X,y,RF,dates,datesDF)
result2 = qimbs.get_performance(Signals,df,dates,0)
result2['I'] = result2.index
ggplot(result2, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>

#LR
from sklearn.linear_model import LogisticRegression as LR
Signals =  qimbs.get_signals1(imbalanceMsg,X,y,LR,dates,datesDF)
result1 = qimbs.get_performance(Signals,df,dates,0)
result1['I'] = result1.index
ggplot(result1, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result1.Pnl.sum()) + geom_line()

# <codecell>

#SVC
from sklearn.svm import SVC
Signals =  qimbs.get_signals1(imbalanceMsg, X[fi['Feature'][:12]],y,SVC,dates,datesDF)
result3 = qimbs.get_performance(Signals,df,dates,0)
result3['I'] = result3.index
ggplot(result1, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result3.Pnl.sum()) + geom_line()

# <codecell>

from IPython.core.display import Image 
Image(filename='/home/user/PyProjects/Results/1.png') 

# <codecell>

#RandomForest
Signals =  qimbs.get_signals_clf(imbalanceMsg,X_neg,y_neg,clf,dates,datesDF_neg)
result2 = qimbs.get_performance(Signals,df,dates,0)
result2['I'] = result2.index
ggplot(result2, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>


























#Regression

# <codecell>

test_size = 20
r = range(len(dates))
np.random.shuffle(r)
test_days = r[:test_size] 
train_days = r[test_size:] 

Xtrain = X_pos.ix[datesDF_pos.ix[train_days],:]
Xtest = X_pos.ix[datesDF_pos.ix[test_days],:]
ytrain = y_pos.ix[datesDF_pos.ix[train_days]]
ytest = y_pos.ix[datesDF_pos.ix[test_days]]

Xtrain.index = range(Xtrain.shape[0])
Xtest.index = range(Xtest.shape[0])
ytrain.index = range(ytrain.shape[0])
ytest.index = range(ytest.shape[0])

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules   import SoftmaxLayer

#Create dataset
#ds = SupervisedDataSet(Xtrain.shape[1], 1)
ds = ClassificationDataSet(Xtrain.shape[1], 1, nb_classes=2)
for j in range(Xtrain.shape[0]):
    ds.addSample(Xtrain.ix[j,:], ytrain.ix[j,:])
ds._convertToOneOfMany( )

#Create net
#from pybrain.structure import FeedForwardNetwork
#net = FeedForwardNetwork()
#from pybrain.structure import LinearLayer, SigmoidLayer
#inLayer = LinearLayer(ds.indim)
#hiddenLayer = SigmoidLayer(5)
#outLayer = SoftmaxLayer(ds.outdim)
#net.addInputModule(inLayer)
#net.addModule(hiddenLayer)
#net.addOutputModule(outLayer)
#from pybrain.structure import FullConnection
#in_to_hidden = FullConnection(inLayer, hiddenLayer)
#hidden_to_out = FullConnection(hiddenLayer, outLayer)
#net.addConnection(in_to_hidden)
#net.addConnection(hidden_to_out)
#net.sortModules()
#net = buildNetwork(ds.indim, 2, ds.outdim, outclass=SoftmaxLayer)

from pybrain.structure import RecurrentNetwork
net = RecurrentNetwork()
net.addInputModule(LinearLayer(ds.indim, name='inLayer'))
net.addModule(SigmoidLayer(ds.indim, name='hiddenLayer'))
net.addOutputModule(SoftmaxLayer(ds.outdim, name='outLayer'))
net.addConnection(FullConnection(net['inLayer'], net['hiddenLayer'], name='in_to_hidden'))
net.addConnection(FullConnection(net['hiddenLayer'], net['outLayer'], name='hidden_to_out'))
net.addRecurrentConnection(FullConnection(net['hiddenLayer'], net['hiddenLayer'], name='hidden_to_hidden'))
net.sortModules()

#Train net
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, momentum=0.1, verbose=True, weightdecay=0.01)

#for i in range(10):
#    if i%20==0:
#        print i
#    trainer.trainEpochs(1)
    
trnerr,testerr = trainer.trainUntilConvergence(dataset=ds,maxEpochs=10)
plt.plot(trnerr,'b',valerr,'r')

# <codecell>

print net.activate(Xtest.ix[1,:])
print ytest.ix[1,:]

# <codecell>

to_hidden=numpy.dot(in_to_hidden.params.reshape(hiddenLayer.dim,inLayer.dim),Xtest.ix[0,:].as_matrix())

# <codecell>

to_out=hiddenLayer.activate(to_hidden)

# <codecell>

in_to_hidden.params.reshape(hiddenLayer.dim,inLayer.dim)

# <codecell>

outLayer.activate(numpy.dot(hidden_to_out.params.reshape(outLayer.dim,hiddenLayer.dim),to_out))

# <codecell>

Xtrain_new= zeros(Xtrain.shape,float)

# <codecell>

hiddenLayer.dim

# <codecell>

def run_reg(X,y,clf_class,n_folds,test_size,dates,datesDF):
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    from sklearn.ensemble import RandomForestRegressor as RFR
    CLF_BEST = clf_class()
    TEST_R2_Score = 0
    
    for i in range(n_folds): 
        #======Get test_train_split=============
        r = range(len(dates))
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
        #======================================== 
        
        f = ytrain<0
        f2 = ytest<0
        
        if (type(clf_class()) ==  type(RFR())):
            clf = clf_class(min_samples_split = ytrain[f].shape[0]*0.05)
        if (type(clf_class()) ==  type(GBR())):
            clf = clf_class(loss='huber', min_samples_split = ytrain[f].shape[0]*0.05,init='zero')    
            
        clf.fit(Xtrain[f], ytrain[f])
        y_predR = clf.predict(Xtest[f2])
        r2 = metrics.r2_score(clf.predict(Xtest[f2]), ytest[f2])
        
        #print r2
           
        if r2>TEST_R2_Score:
            TEST_R2_Score =  r2
            CLF_BEST = clf
                
    print "max R2_Score ",TEST_R2_Score
    
    return CLF_BEST, TEST_R2_Score

# <codecell>

from sklearn import metrics

test_size = 30
r = range(len(dates))
np.random.shuffle(r)
test_days = r[:test_size] 
train_days = r[test_size:] 

Xtrain = X_pos.ix[datesDF_pos.ix[train_days],:]
Xtest = X_pos.ix[datesDF_pos.ix[test_days],:]
ytrain = ypln_pos.ix[datesDF_pos.ix[train_days]]
ytest = ypln_pos.ix[datesDF_pos.ix[test_days]]

from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinR
from sklearn.linear_model import SGDRegressor as SGDR
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

f = ytrain<0
f2 = ytest<0
clf = GBR(loss='huber', min_samples_split = ytrain[f].shape[0]*0.05,init='zero')
clf.fit(Xtrain[f], ytrain[f])
y_pred = clf.predict(Xtest[f2])
print "GBR"
#print "Training mse: ", metrics.mean_squared_error(clf.predict(Xtrain), ytrain)
#print "Test     mse: ", metrics.mean_squared_error(clf.predict(Xtest), ytest)
print "Training mae: ", metrics.mean_absolute_error(clf.predict(Xtrain[f]), ytrain[f])
print "Test     mae: ", metrics.mean_absolute_error(clf.predict(Xtest[f2]), ytest[f2])
print "Training EV: ", metrics.explained_variance_score(clf.predict(Xtrain[f]), ytrain[f])
print "Test     EV: ", metrics.explained_variance_score(clf.predict(Xtest[f2]), ytest[f2])
print "Training R2: ", metrics.r2_score(clf.predict(Xtrain[f]), ytrain[f])
print "Test     R2: ", metrics.r2_score(clf.predict(Xtest[f2]), ytest[f2])
print "-------------"

clf = LinR()#RFR(min_samples_split = ytrain[f].shape[0]*0.05)
clf.fit(Xtrain[f], ytrain[f])
y_predLR = clf.predict(Xtest[f2])
print "LinReg"
#plt.plot(clf.coef_[:10])
#print "Training mse: ", metrics.mean_squared_error(clf.predict(Xtrain), ytrain)
#print "Test     mse: ", metrics.mean_squared_error(clf.predict(Xtest), ytest)
print "Training mae: ", metrics.mean_absolute_error(clf.predict(Xtrain[f]), ytrain[f])
print "Test     mae: ", metrics.mean_absolute_error(clf.predict(Xtest[f2]), ytest[f2])
print "Training EV: ", metrics.explained_variance_score(clf.predict(Xtrain[f]), ytrain[f])
print "Test     EV: ", metrics.explained_variance_score(clf.predict(Xtest[f2]), ytest[f2])
print "Training R2: ", metrics.r2_score(clf.predict(Xtrain[f]), ytrain[f])
print "Test     R2: ", metrics.r2_score(clf.predict(Xtest[f2]), ytest[f2])
print "-------------"

clf = RFR(min_samples_split = ytrain[f].shape[0]*0.05)
clf.fit(Xtrain[f], ytrain[f])
y_predR = clf.predict(Xtest[f2])
print "RF"
print "Training mae: ", metrics.mean_absolute_error(clf.predict(Xtrain[f]), ytrain[f])
print "Test     mae: ", metrics.mean_absolute_error(clf.predict(Xtest[f2]), ytest[f2])
print "Training EV: ", metrics.explained_variance_score(clf.predict(Xtrain[f]), ytrain[f])
print "Test     EV: ", metrics.explained_variance_score(clf.predict(Xtest[f2]), ytest[f2])
print "Training R2: ", metrics.r2_score(clf.predict(Xtrain[f]), ytrain[f])
print "Test     R2: ", metrics.r2_score(clf.predict(Xtest[f2]), ytest[f2])
print "-------------"

rdf=pd.DataFrame()
rdf['y'] = ytest[f2];
rdf['yp'] = y_pred;
rdf['yr'] = y_predR;
rdf['yl'] = y_predLR;
rdf['yall'] = (y_predR+y_pred)/2;
rdf['i'] = range(rdf.shape[0]);
rdf['resid'] = rdf['y']-rdf['yr'];
print "Test     mae: ", metrics.mean_absolute_error(rdf['yall'], rdf['y'])
print "Test     EV: ", metrics.explained_variance_score(rdf['yall'], rdf['y'])
print "Test     R2: ", metrics.r2_score(rdf['yall'], rdf['y'])
print "-------------"


ggplot(rdf,aes('y','yp')) + geom_point(size=1)+\
stat_function(fun = lambda x: x, color='red') +\
geom_point(rdf,aes('y','yl'),size=1,color='green') #+\
#geom_point(rdf,aes('y','yl'),size=1,color='red')# +\

#ggplot(rdf,aes('resid')) + geom_histogram(binwidth = 0.01)

# <codecell>

clf = LinR()
clf.fit(Xtrain[f], ytrain[f])
list(clf.coef_)

# <codecell>

Xtrain.ix[0,:]

# <codecell>

reload(qimbs)

# <codecell>

#Regression learning curves

# <codecell>

qimbs.Regression('COMB', X_pos,ypln_pos,dates,datesDF_pos,-1)

# <codecell>

qimbs.Regression('COMB', X_neg,ypln_neg,dates,datesDF_neg,-1)

# <codecell>

qimbs.Regression('COMB', X_neg,ypln_neg,dates,datesDF_neg,1)

# <codecell>

qimbs.Regression('COMB', X_pos,ypln_pos,dates,datesDF_pos,1)

# <codecell>

f = ytrain<0
f2 = ytest<0
clf = GradientBoostingRegressor(loss='huber', min_samples_split = ytrain[f].shape[0]*0.05)
clf.fit(Xtrain[f], ytrain[f])
fix, axs = pdep.plot_partial_dependence(clf,Xtrain[f],[(5,0)],feature_names=Xtrain.columns)

# <codecell>

f = ytrain>0
f2 = ytest>0
clf = GradientBoostingRegressor(loss='huber', min_samples_split = ytrain[f].shape[0]*0.05)
clf.fit(Xtrain[f], ytrain[f])
fix, axs = pdep.plot_partial_dependence(clf,Xtrain[f],[(5,0)],feature_names=Xtrain.columns)

# <codecell>

tmp_df = pd.DataFrame()
tmp_df['yp'] = ypln_pos
tmp_df['yn'] = ypln_neg
tmp_df['t'] = X_pos.D555
tmp_df['t2'] = X_neg.D444
print np.percentile(tmp_df[(tmp_df.yp>0) & (tmp_df.t>=0)].yp,95)
print np.percentile(tmp_df[(tmp_df.yn>0) & (tmp_df.t2>=0)].yp,95)
ggplot(tmp_df[(tmp_df.yp>0) & (tmp_df.t>=0)],aes(x='yp')) + geom_histogram(binwidth = 0.01,alpha=0.5, color='red') +\
geom_histogram(tmp_df[(tmp_df.yn>0) & (tmp_df.t2>=0)],aes(x='yn'),binwidth = 0.01,alpha=0.5,color='green')


# <codecell>

tmp_df = pd.DataFrame()
tmp_df['yp'] = ypln_pos
tmp_df['yn'] = ypln_neg
tmp_df['t'] = X_pos.D555
tmp_df['t2'] = X_neg.D444
ggplot(tmp_df[(tmp_df.yp<0) & (tmp_df.t>=0)],aes(x='yp')) + geom_histogram(binwidth = 0.005,alpha=0.5, color='red') +\
geom_histogram(tmp_df[(tmp_df.yn<0) & (tmp_df.t2>=0)],aes(x='yn'),binwidth = 0.005,alpha=0.5,color='green')

# <codecell>

X_pos.columns

# <codecell>

tmp_df = X_pos.copy()
tmp_df['yp'] = ypln_pos
tmp_df['NearN'] =  np.exp(np.exp(tmp_df.Near))
ggplot(tmp_df,aes(x='a7',y='yp')) + geom_point(size=1)

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

def Forest2SqlClf(advantage,db,fdf,X_pos,X_neg,dates,datesDF_pos,datesDF_neg):

    ym_pos = fdf.OPC_P.copy()
    ym_pos[:] = 0
    ym_pos[fdf.OPC_P>fdf.Ask_P+advantage/100] = 1
    ym_pos = ym_pos[fdf.a14>0]
    ym_pos.index = range(y_pos.shape[0])

    ym_neg = fdf.OPC_P.copy()
    ym_neg[:] = 0
    ym_neg[fdf.OPC_P<fdf.Bid_P-advantage/100] = 1
    ym_neg = ym_neg[fdf.a14<0]
    ym_neg.index = range(y_neg.shape[0])
    
    trainError, testError, cm, clf, fscore1 = qimbs.run_cv_proba(X_pos,ym_pos,RF,20,20,dates,datesDF_pos)
    #clf = RF(min_samples_split = X_pos.shape[0]*0.05, criterion = 'entropy')
    #clf.fit(X_pos,ym_pos)
    qimbs.Forest2Sql(clf, X_pos,0,1,advantage,fscore1,db)
    
    trainError, testError, cm, clf, fscore2 = qimbs.run_cv_proba(X_neg,ym_neg,RF,20,20,dates,datesDF_neg)
    #clf = RF(min_samples_split = X_neg.shape[0]*0.05, criterion = 'entropy')
    #clf.fit(X_neg,ym_neg)
    qimbs.Forest2Sql(clf,X_neg,0,-1,advantage,fscore2,db)
    
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

# <codecell>

#Random forest for regression


# <codecell>


# <codecell>


# <codecell>

#Random forest for classification
from sklearn.ensemble import RandomForestClassifier as RF
clf = RF(min_samples_split = X_pos.shape[0]*0.05, criterion = 'entropy')
clf.fit(X_pos,y_pos)
qimbs.Forest2Txt(clf, X_pos.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/PosNotFirst')

clf = RF(min_samples_split = X_neg.shape[0]*0.05, criterion = 'entropy')
clf.fit(X_neg,y_neg)
qimbs.Forest2Txt(clf, X_neg.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/NegNotFirst')

# <codecell>

#Random forest for classification far move
clf = RF(n_jobs=4,min_samples_split = X_pos.shape[0]*0.05, criterion = 'entropy')
clf.fit(X_pos,ym_pos)
qimbs.Forest2Txt(clf, X_pos.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/5Pos')
clf = RF(min_samples_split = X_neg.shape[0]*0.05, criterion = 'entropy')
clf.fit(X_neg,ym_neg)
qimbs.Forest2Txt(clf, X_neg.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/5Neg')

# <codecell>

#Gradient Boosting for classification far move
clf = GBC(min_samples_split = X_pos.shape[0]*0.05, init='zero')
clf.fit(X_pos,ym_pos)
#qimbs.Forest2Txt(clf, X_pos.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/5Pos')
clf = GBC(min_samples_split = X_neg.shape[0]*0.05, init='zero')
clf.fit(X_neg,ym_neg)
#qimbs.Forest2Txt(clf, X_neg.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/5Neg')

# <codecell>

#Random forest for classification
clf = RF(min_samples_split = X_pos.shape[0]*0.05, criterion = 'entropy')
clf.fit(X_pos,y_pos)
qimbs.Forest2Txt(clf, X_pos.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/PosLast')
clf = RF(min_samples_split = X_neg.shape[0]*0.05, criterion = 'entropy')
clf.fit(X_neg,y_neg)
qimbs.Forest2Txt(clf, X_neg.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/NegLast')

# <codecell>


# <codecell>

#Gradient Boosting for classification
clf = GBC(min_samples_split = X_pos.shape[0]*0.05, init='zero')
clf.fit(X_pos,y_pos)
qimbs.Forest2Txt(clf, X_pos.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/Pos')
clf = GBC(min_samples_split = X_neg.shape[0]*0.05, init='zero')
clf.fit(X_neg,y_neg)
qimbs.Forest2Txt(clf, X_neg.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/Neg')

# <codecell>

#Random forest for regression
from sklearn.ensemble import RandomForestRegressor as RFR
Xpp = X_pos.copy()
Xpp = Xpp[ypln_pos>0]
Xpp.index = range(Xpp.shape[0])

clf = RFR(min_samples_split = ypln_pos[ypln_pos>0].shape[0]*0.05)
clf.fit(Xpp, ypln_pos[ypln_pos>0])
qimbs.Forest2Txt(clf, Xpp.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/PP')

clf = GBR(min_samples_split = ypln_pos[ypln_pos>0].shape[0]*0.05, loss='huber',init='zero',learning_rate=0.1)
clf.fit(Xpp, ypln_pos[ypln_pos>0])
qimbs.Forest2Txt(clf, Xpp.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/PP')

# <codecell>

Xpn = X_pos.copy()
Xpn = Xpn[ypln_pos<0]
Xpn.index = range(Xpn.shape[0])

clf = RFR(min_samples_split = ypln_pos[ypln_pos<0].shape[0]*0.05)
clf.fit(Xpn, ypln_pos[ypln_pos<0])
qimbs.Forest2Txt(clf, Xpn.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/PN')

clf = GBR(min_samples_split = ypln_pos[ypln_pos<0].shape[0]*0.05, loss='huber',init='zero',learning_rate=0.1)
clf.fit(Xpn, ypln_pos[ypln_pos<0])
qimbs.Forest2Txt(clf, Xpn.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/PN')

# <codecell>

Xnp = X_neg.copy()
Xnp = Xnp[ypln_neg>0]
Xnp.index = range(Xnp.shape[0])

clf = RFR(min_samples_split = ypln_neg[ypln_neg>0].shape[0]*0.05)
clf.fit(Xnp, ypln_neg[ypln_neg>0])
qimbs.Forest2Txt(clf, Xnp.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/NP')

clf = GBR(min_samples_split = ypln_neg[ypln_neg>0].shape[0]*0.05, loss='huber',init='zero',learning_rate=0.1)
clf.fit(Xnp, ypln_neg[ypln_neg>0])
qimbs.Forest2Txt(clf, Xnp.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/NP')

# <codecell>

Xnn = X_neg.copy()
Xnn = Xnn[ypln_neg<0]
Xnn.index = range(Xnn.shape[0])

clf = RFR(min_samples_split = ypln_neg[ypln_neg<0].shape[0]*0.05)
clf.fit(Xnn, ypln_neg[ypln_neg<0])
qimbs.Forest2Txt(clf, Xnn.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/NN')

clf = GBR(min_samples_split = ypln_neg[ypln_neg<0].shape[0]*0.05, loss='huber',init='zero',learning_rate=0.1)
clf.fit(Xnn, ypln_neg[ypln_neg<0])
qimbs.Forest2Txt(clf, Xnn.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/NN')

# <codecell>

reload(qimbs)

# <codecell>

from sklearn.ensemble import RandomForestClassifier as RF
clf = RF(min_samples_split = X_pos.shape[0]*0.05, criterion = 'entropy')
clf.fit(X_pos,y_pos)

# <codecell>

import sqlite3 as lite
import sys
import pandas as pd
con = lite.connect(db)
df = pd.read_sql("SELECT * FROM RFnames", con)
print df
pd.read_sql("SELECT * FROM RFtests Order by RFID",con)
con.commit()
con.interrupt()
con.close()

# <codecell>

clfPos = RF(min_samples_split = X_pos.shape[0]*0.05, criterion = 'entropy')
clfPos.fit(X_pos,ym_pos)
clfMin = RF(min_samples_split = X_neg.shape[0]*0.05, criterion = 'entropy')
clfMin.fit(X_neg,ym_neg)
predPos = pd.DataFrame(clfPos.predict(X_pos),columns=['val'])
predNeg = pd.DataFrame(clfMin.predict(X_neg),columns=['val'])

