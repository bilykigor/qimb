# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import numpy as np
import pandas as pd
from ggplot import *
from sklearn.preprocessing import PolynomialFeatures
import os
from scipy.cluster import *
from scipy.spatial.distance import cdist

# <codecell>

driversFiles = map(int,os.listdir('../Kaggle/drivers/'))
driversFiles.sort(reverse=False)
speedDir = '../Kaggle/features/speed'
caccDir = '../Kaggle/features/cacc'

# <codecell>

''' Submission 3'''

def driver_trip(x):
    return str(int(x[1]))+'_'+str(int(x[2]))
def prob(x):
    return int(sum(x[10:])!=0)
def norm(x):
    return x/sum(x)
def myDist (x,y):
    res = (x-y)**2
    res = res * normVector
    return np.sqrt(res.sum())

mySubmission = pd.DataFrame(columns=['driver_trip','prob'])
for driver in driversFiles:
    df = pd.read_csv(speedDir+'/' + str(driver) + '.csv')
    
    data = df.ix[:,3:]
    normVector = data.median()
    normVector = 1-array(normVector/sum(normVector))
    normVector = sqrt(normVector)
    data = data.apply(norm,axis=1)
    
    centres, xtoc, dist = kmeanssample(array(data), 2, nsample=1,
            delta=data.mean().mean()/1000, maxiter=1000, metric=myDist, verbose=0 )
    
    df['prob'] = 1
    df.prob[xtoc==Counter(xtoc).keys()[argmin(Counter(xtoc).values())]] = 0
    df['driver_trip'] = df.apply(driver_trip,axis=1)
    
    mySubmission=mySubmission.append(df[['driver_trip','prob']])
    #print driver
mySubmission.index = range(mySubmission.shape[0])
mySubmission.to_csv('../Kaggle/mySubmission.csv',  index_col=None)

# <codecell>

''' Submission 5 parallel'''
def driver_trip(x):
    return str(int(x[1]))+'_'+str(int(x[2]))
def prob(x):
    return int(sum(x[10:])!=0)
def norm(x):
    return x/sum(x)
def myDist (x,y):
    res = (x-y)**2
    res = res * normVector
    return np.sqrt(res.sum())

def GetFeatures(driver,core):
    df = pd.read_csv(speedDir+'/' + str(driver) + '.csv')
    
    data = df.ix[:,3:]
    normVector = data.median()
    normVector = 1-array(normVector/sum(normVector))
    normVector = sqrt(normVector)
    data = data.apply(norm,axis=1)
    kmdelta = data.mean().mean()/1000
    
    error=inf
    for i in range(2,20):
        centres, xtoc, dist = kmeanssample(array(data), i, nsample=1,delta = kmdelta, maxiter=1000, metric=myDist, verbose=0 )
        #print i, sum(dist)
        if sum(dist)>error:
            centres, xtoc, dist = kmeanssample(array(data), i-1, nsample=1,delta = kmdelta, maxiter=1000, metric=myDist, verbose=0 )
            break
        error =  sum(dist)
        
    df['prob'] = 1
    df.prob[xtoc==Counter(xtoc).keys()[argmin(Counter(xtoc).values())]] = 0
    df['driver_trip'] = df.apply(driver_trip,axis=1)
    
    df[['driver_trip','prob']].to_csv('../Kaggle/' + str(core) + '.csv')
    return 0 

import time
from joblib import Parallel, delayed  
import multiprocessing
num_cores=multiprocessing.cpu_count()

mySubmission = pd.DataFrame(columns=['driver_trip','prob'])
inputs = range(int(len(driversFiles)/4))
for i in inputs:
    r=range(4*i,4*i+4)
    Parallel(n_jobs=num_cores)(delayed(GetFeatures)(driversFiles[j],j% 4) for j in r)
    mySubmission=mySubmission.append(pd.read_csv('../Kaggle/0.csv'))
    mySubmission=mySubmission.append(pd.read_csv('../Kaggle/1.csv'))
    mySubmission=mySubmission.append(pd.read_csv('../Kaggle/2.csv'))
    mySubmission=mySubmission.append(pd.read_csv('../Kaggle/3.csv'))
    #break
mySubmission.index = range(mySubmission.shape[0])
mySubmission.to_csv('../Kaggle/mySubmission.csv',  index_col=None)

# <codecell>


