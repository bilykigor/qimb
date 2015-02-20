# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
from ggplot import *
import os
from Trip import Trip
import seaborn as sns
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

# <codecell>

driversFiles = map(int,os.listdir('/home/user1/Desktop/SharedFolder/Kaggle/DriversOriginal/'))
randomDrivers = map(int,os.listdir('/home/user1/Desktop/SharedFolder/Kaggle/DriversOriginal/'))
driversFiles.sort(reverse=False)

# <codecell>

tripFiles = range(1,201)
selectedCols = ['v','acc','acc2']

# <codecell>

def GetGmm(xN):
    n_components = [24]#range(20,25)
    gmms = [GMM(n_components=n, covariance_type='full').fit(xN) for n in n_components]
    BICs = [gmm.bic(xN) for gmm in gmms]
    i_min = np.argmin(BICs)
    clf=gmms[i_min]
    #print '%s components - BIC %s' %(n_components[i_min],BICs[i_min])
    #tol = np.percentile(np.exp(clf.score(xN)),10)
    return clf

# <codecell>

def SaveGmm(clf):
    cos = pd.DataFrame(np.zeros((clf.n_components,6)))
    for i in range(clf.n_components):
        cos.ix[i,:1] = clf.means_[i]
        cos.ix[i,2:] = array(clf.covars_[i]).flatten()
    return cos

# <codecell>

vlim=[0.01,40]
clim=[-20,20]
def GetFeatures(driverID,j):
    driverDir = '/home/user1/Desktop/SharedFolder/Kaggle/DriversCleaned/'+str(driverID)

    tripFiles = range(1,201)
   
    
    X = pd.DataFrame(columns=selectedCols)
    for index,tripID in enumerate(tripFiles):       
        #print index,tripID
        trip = Trip(driverID,tripID,pd.read_csv(driverDir+'_' + str(tripID) + '.csv'))
        trip.getSpeed()
        trip.getAcc()
        #trip.getRadius()
        #trip.getCacc()
        trip.getFeatures()
        
        '''z=array(list(set(np.asarray([range(x-5,x+5) for x in (trip.features.v<vlim[0]).nonzero()[0]]).flatten())))
        z=z[z<trip.features.shape[0]]
        z=z[z>=0]
        #z=array(list(set(range(trip.features.shape[0]))-set(z)))
    
        Xz=trip.features.loc[z]
        Xz=Xz.reset_index(drop=True)

        Xz=Xz.loc[Xz.v!=0]
        Xz=Xz.reset_index(drop=True)

        X = X.append(Xz)'''
        X = X.append(trip.features)
        
    X=X.reset_index(drop=True) 
    
    X=X[(X.v<vlim[1]) & (X.v>vlim[0])]
    X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
    X=X.reset_index(drop=True) 
    
    clf=GetGmm(np.asanyarray(X[['v','acc']]))
    cos = SaveGmm(clf)
    
    cos.to_csv('/home/user1/Desktop/SharedFolder/Kaggle/FeaturesCleaned/GMM/All/' + str(driverID) + '.csv', index=False)
    #del cos

    return 0

# <codecell>

i=i+1
r=range(num_cores*i,num_cores*i+num_cores)

# <codecell>

i

# <codecell>

[GetFeatures(driversFiles[j],j)  for j in r]

# <codecell>

Parallel(n_jobs=num_cores)(delayed(MakePrediction)(driversFiles[j],j % num_cores) for j in r) 

# <codecell>

import time
from joblib import Parallel, delayed  
import multiprocessing
num_cores=multiprocessing.cpu_count()

'''for driverID in driversFiles:
    print driverID
    MakePrediction(driverID,1)
'''
num_cores=4
inputs = range(470,len(driversFiles)/num_cores)
for i in inputs:
    start = time.time()
    r=range(num_cores*i,num_cores*i+num_cores)
    Parallel(n_jobs=num_cores)(delayed(MakePrediction)(driversFiles[j],j % num_cores) for j in r) 
    #print time.time()-start 

# <codecell>

def GetProba(drive,driverID,tripInd):
    driverDir = '/home/user1/Desktop/SharedFolder/Kaggle/DriversCleaned/'+str(driverID)
    df = pd.read_csv(driverDir+'_' + str(tripInd)+'.csv')
    trip = Trip(driverID,tripInd,df)
    trip.getSpeed()
    trip.getAcc()
    #trip.getRadius()
    #trip.getCacc()
    trip.getFeatures()
    X=trip.features[['v','acc']]
    
    probas = np.zeros((X.shape[0],drive.shape[0]))
    for i in range(drive.shape[0]):
        probas[:,i]=multivariate_normal.pdf(X, mean=array(drive.ix[i,:2]), cov=[array(drive.ix[i,2:4]),array(drive.ix[i,4:])])

    probas=np.max(probas,axis=1)
    return probas.mean()

# <codecell>

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
import EnsembleClassifier as eclf
reload(eclf)

randTrips = range(1,200)
drivers_sampleSize=5
trips_sampleSize=170

def MakePrediction(cur_driver,tk):
    #print cur_driver
    drive=pd.read_csv('/home/user1/Desktop/SharedFolder/Kaggle/FeaturesCleaned/GMM/All/' + str(cur_driver) + '.csv')
    Xpred = pd.DataFrame(np.zeros((200,1)),columns=['val'])
    for i in range(1,201):
        Xpred.loc[i-1]=GetProba(drive,cur_driver,i)
    
    if len(pd.isnull(Xpred).any(1).nonzero()[0])>0:
        print 'error'
    
    cur_driver_df = pd.DataFrame(np.zeros((Xpred.shape[0],3)),columns=['driver','trip','prob'])
    cur_driver_df.driver = cur_driver
    cur_driver_df.trip = range(1,201)
    
    np.random.shuffle(randTrips)  
    Xtrain = Xpred.loc[randTrips[:trips_sampleSize]]
        
    ytrain = pd.DataFrame(np.ones((trips_sampleSize,1)))
    ytrain.ix[:] = int(1)
    
    np.random.shuffle(randomDrivers)           
    for rand_driver in randomDrivers[:drivers_sampleSize]:
        if rand_driver==cur_driver:
            continue
        drive=pd.read_csv('/home/user1/Desktop/SharedFolder/Kaggle/FeaturesCleaned/GMM/All/' + str(rand_driver) + '.csv')
        D = pd.DataFrame(np.zeros((trips_sampleSize/drivers_sampleSize,1)),columns=['val'])
        
        np.random.shuffle(randTrips)
        for i in range(trips_sampleSize/drivers_sampleSize):
            D.loc[i]=GetProba(drive,rand_driver,randTrips[i])
            
        Xtrain=Xtrain.append(D)
              
        tmp = pd.DataFrame(np.zeros((trips_sampleSize/drivers_sampleSize,1)))
        tmp.ix[:] = int(0)
        ytrain = ytrain.append(tmp)  
        
        if len(pd.isnull(D).any(1).nonzero()[0])>0:
            print 'error'
        
    Xtrain=Xtrain.reset_index(drop=True)
    ytrain=ytrain.reset_index(drop=True)
        
    #preprocessing
    #pca.fit(Xtrain)
    
    
    #fit model  
    #===============================================================================
    clf = \
    eclf.EnsembleClassifier(
    clfs=[
    LR(class_weight='auto',C=0.5)
    ,RF()
    ,GBC()
    ])
    
    tmpInd = pd.isnull(Xtrain).any(1).nonzero()[0]
    if (len(tmpInd)>0):
        print Xtrain.loc[tmpInd]
    clf.fit(Xtrain,np.asarray(ytrain[0]))
    #===============================================================================
    cur_driver_df.prob=clf.predict_proba(Xpred)[:,1]
    
    cur_driver_df.to_csv('/home/user1/Desktop/SharedFolder/Kaggle/results/' + str(cur_driver) + '.csv', index=False)  
    
    return 0

# <codecell>

MakePrediction(1,1)

# <codecell>

f = lambda x:  '%s_%s' % (int(x[0]),int(x[1]))
main_df = pd.DataFrame(columns=['driver','trip','prob'])
for iter, cur_driver in enumerate(driversFiles):
    cur_driver_df = pd.read_csv('/home/user1/Desktop/SharedFolder/Kaggle/results/' + str(cur_driver) + '.csv')

    if (cur_driver in set(main_df.driver)):
        print 'Error'
        break
        
    main_df=main_df.append(cur_driver_df)
        
main_df.index = range(main_df.shape[0])
main_df['driver_trip'] = main_df.apply(f,axis=1)
main_df[['driver_trip','prob']].to_csv('/home/user1/Desktop/SharedFolder/Kaggle/gmm.csv',  index=False)

# <codecell>

main_df.head()

# <codecell>


