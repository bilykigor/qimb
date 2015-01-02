# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np

# <codecell>

indata = pd.read_csv("indata", delimiter='  ',header = None, names = ['x1','x2','y'])
outdata = pd.read_csv("outdata", delimiter='  ',header = None, names = ['x1','x2','y'])

# <codecell>

def transform(data,k):
    result = data[['x1','x2']].copy()
    result['c'] = 1
    if k>=3:
        result['x1x1'] = data.x1*data.x1
    if k>=4:
        result['x2x2'] = data.x2*data.x2
    if k>=5:
        result['x1x2'] = data.x1*data.x2
    if k>=6:
        result['x1-x2'] = abs(data.x1-data.x2)
    if k>=7:
        result['x1+x2'] = abs(data.x1+data.x2)
    return result

# <codecell>

from sklearn.linear_model import LinearRegression as LR

for k in range(3,8):
    clf = LR()
    traindata = transform(indata,k).ix[:24,:]
    ytrain = indata.y.ix[:24]
    #valdata = transform(indata,k).ix[10:,:]
    #yval = indata.y.ix[10:]
    valdata = transform(outdata,k)
    yval = outdata.y
    clf.fit(traindata,ytrain)
    error = float((np.sign(clf.predict(valdata))!=yval).sum())/len(yval)
    print k, error

# <codecell>

#ro = pow(pow(3,0.5)+4,0.5)
#ro = pow(pow(3,0.5)-1,0.5)
ro = pow(9+4*pow(6,0.5),0.5)
df= pd.DataFrame(columns=['x','y'])
df=df.append({'x':-1,'y':0},ignore_index=True)
df=df.append({'x':ro,'y':1},ignore_index=True)
df=df.append({'x':1,'y':0},ignore_index=True)

# <codecell>

(pow(df.ix[[0,1],:].y.mean() - df.ix[2].y,2)+\
pow(df.ix[[0,2],:].y.mean() - df.ix[1].y,2)+\
pow(df.ix[[1,2],:].y.mean() - df.ix[0].y,2))/3

# <codecell>

(line(df.ix[[0,1],:],df.ix[2,:])+\
line(df.ix[[0,2],:],df.ix[1,:])+\
line(df.ix[[1,2],:],df.ix[0,:]))/3

# <codecell>

def line(d,v):
    d= d.copy()
    d.index = range(2)
    y = lambda x: d.y[0]+(x-d.x[0])*(d.y[1]-d.y[0])/(d.x[1]-d.x[0])
    #print v.x
    return pow(y(v.x) - v.y,2)

# <codecell>

y = lambda x: df.y[1]+(x-df.x[1])*(df.y[2]-df.y[1])/(df.x[2]-df.x[1])

# <codecell>


