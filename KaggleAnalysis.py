# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import numpy as np
import pandas as pd
from ggplot import *
from sklearn.preprocessing import PolynomialFeatures
import os
import Trip

# <codecell>

trips = []

driverID = 1
driverDir = '../Kaggle/drivers/'+str(driverID)
tripFiles = os.listdir(driverDir)

for index, tripFile in enumerate(tripFiles):
    if (tripFile.split('.')[0][-1]!='f'):
        df = pd.read_csv(driverDir+'/' + tripFile)
        trip = Trip(1,int(tripFile.split('.')[0]),df)
        trips.append(trip)  

        if index>2:
            break

# <codecell>

def speedDistribution(trip,window=1):
    x=np.asarray(trip.x)
    y=np.asarray(trip.y)
    vitesse = np.sqrt((x[window:]-x[:-window])**2 + (y[window:]-y[:-window])**2)/window
    return np.percentile(vitesse,[x*5 for x in range(1,20)])

# <codecell>

tripInd = 1
start = 0
end = trips[tripInd].n

# <codecell>

trips[tripInd].ID
#Radius(46)

# <codecell>

trips[tripInd].coordinates.ix[start:end,:]

# <codecell>

trips[tripInd].radius.ix[start:end,:]

# <codecell>

trips[tripInd].speed.ix[start:end,:]

# <codecell>

trips[tripInd].features.ix[start:end,:]

# <codecell>

ggplot(trips[tripInd].coordinates.ix[start:end,:],aes(x='x',y='y')) + geom_point(size=5)

# <codecell>

df = trips[tripInd].radius.copy()
df['ind'] = range(df.shape[0])
ggplot(df.ix[start:end,:],aes(x='ind',y='m')) + geom_point(size=10) 

# <codecell>

df = trips[tripInd].speed.copy()
df['ind'] = range(df.shape[0])
ggplot(df.ix[start:end,:],aes(x='ind',y='m_s')) + geom_point(size=10)

# <codecell>

df = trips[tripInd].features.copy()
df['ind'] = range(df.shape[0])
ggplot(df.ix[start:end,:],aes(x='ind',y='acc')) + geom_point(color='green',size=10)+\
geom_point(df.ix[start:end,:],aes(x='ind',y='cacc'),color='red',size=10)

# <codecell>

ggplot(trips[tripInd].features[trips[tripInd].features.v<=5].ix[start:end,:],aes(x='acc',y='cacc')) + geom_point(size=10)

# <codecell>

ggplot(trips[tripInd].features[trips[tripInd].features.v<=5].ix[start:end,:],aes(x='acc')) + geom_histogram(binwidth = 0.1)

# <codecell>

ggplot(trips[tripInd].features[trips[tripInd].features.v>0].ix[start:end,:],aes(x='v')) + geom_histogram(binwidth = 0.5)

# <codecell>

(n,b,p) = matplotlib.pyplot.hist(list(trips[tripInd].features[trips[tripInd].features.v>=0].acc),bins=50,range=[-5,5])

# <codecell>

(n,b,p) = matplotlib.pyplot.hist(list(trips[tripInd].features[trips[tripInd].features.v>=0].v),bins=50,range=[0,40])

# <codecell>

df = pd.DataFrame(np.zeros((len(b)/2,2)))
df.columns = ['x','y']
df.x = b[len(b)/2+1:]
_n = df.shape[0]
for i in range(_n):
    df.y[i] = (n[_n+i]-n[_n-i-1])/sum(n)
ggplot(df,aes(x='x',y='y')) + geom_point(size=10) + geom_line() + geom_hline(yintercept=0) 

# <codecell>


