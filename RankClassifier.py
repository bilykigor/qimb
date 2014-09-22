# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import numpy as np
import pandas as pd
from ggplot import *
from sklearn.ensemble import RandomForestClassifier as RF

# <codecell>

reload(qimbs)

# <codecell>

#!Importing data
df = pd.DataFrame()
for i in range(5,7):
    df = df.append(qimbs.import_month2(i))
    print i
print df.shape
df.index = range(df.shape[0])

#Adding Timestamp column
#df = qimbs.create_timestamp(df)

# <codecell>

#Getting imbalance info
imbalanceMsg = qimbs.get_imbelanceMSG3(df,0)
imbalanceMsg = imbalanceMsg[
    (imbalanceMsg.nsdq_AP - imbalanceMsg.nsdq_BP < 
     1.0 * (imbalanceMsg.nsdq_AP + imbalanceMsg.nsdq_BP)*0.5) ]
imbalanceMsg.index = range(imbalanceMsg.shape[0])
imbalanceMsg.shape

# <codecell>

#Creating features
fdf,Features = qimbs.create_features4(imbalanceMsg)

# <codecell>

X = fdf[['1','2','3','n2','n3','2_2','3_2','4','5','6', '7', '8']]#,'9','10','11','12','13','14','15','16','17','18',
         #'19','20','21','22','23','24','25','26']]

y = fdf['Move']

dates = sorted(list(set(fdf.Date)))
datesDF = qimbs.dates_tmp_df(fdf)

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

#Get ranks
X=np.argsort(X, axis=1)
X['0'] = fdf['0']+1

import sklearn.preprocessing as preprocessing
XM = pd.DataFrame(preprocessing.OneHotEncoder(sparse = False, dtype=np.int).fit_transform(X))

# <codecell>

#Apply Random Forest
qimbs.OneModelResults(RF,XM,y,ERRORS,dates,datesDF)

# <codecell>

#RandomForest
Signals =  qimbs.get_signals1(imbalanceMsg,XM,y,RF,dates,datesDF)
result2 = qimbs.get_performance(Signals,df,dates,0)
result2['I'] = result2.index
ggplot(result2, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>

#Apply Random Forest
qimbs.OneModelResults(RF,XM,y,ERRORS,dates,datesDF)

# <codecell>

#RandomForest
Signals =  qimbs.get_signals1(imbalanceMsg,XM,y,RF,dates,datesDF)
result2 = qimbs.get_performance(Signals,df,dates,0)
result2['I'] = result2.index
ggplot(result2, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>

#Apply Random Forest
qimbs.OneModelResults(RF,XM,y,ERRORS,dates,datesDF)

# <codecell>

#RandomForest
Signals =  qimbs.get_signals1(imbalanceMsg,XM,y,RF,dates,datesDF)
result2 = qimbs.get_performance(Signals,df,dates,0)
result2['I'] = result2.index
ggplot(result2, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>

#Apply Random Forest
qimbs.OneModelResults(RF,XM,y,ERRORS,dates,datesDF)

# <codecell>

#RandomForest
Signals =  qimbs.get_signals1(imbalanceMsg,XM,y,RF,dates,datesDF)
result2 = qimbs.get_performance(Signals,df,dates,0)
result2['I'] = result2.index
ggplot(result2, aes('I','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>

#Apply Random Forest
qimbs.OneModelResults(RF,XM,y,ERRORS,dates,datesDF)

# <codecell>

#RandomForest
Signals =  qimbs.get_signals1(imbalanceMsg,XM,y,RF,dates,datesDF)
Symbols = sorted(list(set(df.Symbol)))
SymbolsInd=dict()
for i in range(len(Symbols)):
    SymbolsInd[Symbols[i]]=i
T = zeros((len(Symbols),len(Symbols)))
TN = zeros((len(Symbols),len(Symbols)))
result2 = qimbs.get_performance(Signals,df,dates,SymbolsInd,T,TN,0)
ggplot(result2, aes('Date','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>

#RandomForest
Signals =  qimbs.get_signals_proba(imbalanceMsg,XM,y,RF,dates,datesDF)
Symbols = sorted(list(set(df.Symbol)))
SymbolsInd=dict()
for i in range(len(Symbols)):
    SymbolsInd[Symbols[i]]=i
T = zeros((len(Symbols),len(Symbols)))
TN = zeros((len(Symbols),len(Symbols)))
result2 = qimbs.get_performance(Signals,df,dates,SymbolsInd,T,TN,0)
ggplot(result2, aes('Date','Pnl')) + geom_point() + ggtitle('Sum=%s' % result2.Pnl.sum()) + geom_line()

# <codecell>


