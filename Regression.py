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
df = qimbs.import_month(7)

# <codecell>

#Adding Timestamp column
df = qimbs.create_timestamp(df)

# <codecell>

#Getting imbalance info
imbalanceMsg = qimbs.get_imbelanceMSG(df,0)

# <codecell>

#Creating features
fdf,Features = qimbs.create_features2(imbalanceMsg)

# <codecell>

fdf=fdf[(abs(fdf.Far)<0.2) & (abs(fdf.Near)<0.2) &
        (abs(fdf.PrevCLC)<0.2) & (abs(fdf.PrevOPC)<0.2) &
        (abs(fdf.Bid)<0.2) & (abs(fdf.Ask)<0.2)]
fdf.index = range(fdf.shape[0])

# <codecell>

X = fdf[['Bid', 'Ask',  'Near', 'Far',  'PrevCLC',  'PrevOPC']]
#X['Bias'] = np.ones((X.shape[0],1))

y = fdf['Move']

dates = sorted(list(set(fdf.Date)))
datesDF = qimbs.dates_tmp_df(fdf)

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

# <codecell>

y[y>0]=1
y[y<0]=-1

# <codecell>

ggplot(fdf,aes(x='Move')) + geom_histogram(binwidth = 0.05)

# <codecell>

X.describe()

# <codecell>

axes = pd.tools.plotting.scatter_matrix(X, color="brown")

# <codecell>

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
XN=scaler.transform(X)
X_norm = pd.DataFrame(XN,columns=X.columns)
X_norm.describe()

# <codecell>

ggplot(X_norm,aes(x='PrevCLC')) + geom_histogram(binwidth=0.05)

# <codecell>

for v in X.columns:
    X_norm[v]=qimbs.sigmoid(X_norm[v]*0.5)
X_norm.describe()

# <codecell>

axes = pd.tools.plotting.scatter_matrix(X_norm, color="brown")

# <codecell>

y = X['Move']
X = X[['Bid', 'Ask',  'Near', 'Far', 'PrevOPC', 'PrevCLC']]

# <codecell>

from sklearn.decomposition import PCA
pca = PCA(n_components=7)
pca.fit(XN)
plt.plot(pca.explained_variance_ratio_) 
X_pca=pd.DataFrame(pca.transform(XN),columns=['v1','v2','v3','v4','v5','v6','v7'])
X_pca.describe()

# <codecell>

pca.components_

# <codecell>

ggplot(X[y>=0.5], aes('Near','Far')) + geom_point(size=1, color='green') +\
geom_point(X[y<0.5], aes('Near','Far'),size=1, color='red')

# <codecell>

from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
#r = SVR().fit(X,y)
r = RF(n_jobs=4).fit(X,y)
pred = r.predict(X)
print 'MAE: %s, MSE: %s, R2: %s' % \
(mean_absolute_error(y,pred),mean_squared_error(y,pred),\
 r2_score(y,pred))

# <codecell>

plt.scatter(y,r.predict(X)) #+ geom_abline(color='red')

# <codecell>

dates = sorted(list(set(fdf.Date)))
datesDF = qimbs.dates_tmp_df(fdf)

ERRORS = pd.DataFrame(columns=['Model','TrainError','TestError'])

# <codecell>

#Apply Random Forest
from sklearn.ensemble import RandomForestClassifier as RF
print "Random forest:"
qimbs.OneModelResults(RF, X,y,ERRORS,dates,datesDF,n_ensembles=10, test_size_ensemble=0.2)

# <codecell>


