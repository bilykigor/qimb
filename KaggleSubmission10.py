# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from scipy.stats import multivariate_normal
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

# <codecell>

driversFiles = map(int,os.listdir('../Kaggle/drivers/'))
randomDrivers = map(int,os.listdir('../Kaggle/drivers/'))
driversFiles.sort(reverse=False)

# <codecell>

def GetFeatures(driverID,j):
    driverDir = '/home/user1/Desktop/Share2Windows/Kaggle/DriversCleaned/'+str(driverID)
    tripFiles = range(1,201)
   
    cacc = pd.DataFrame(np.zeros((1,51)))
    for index,tripID in enumerate(tripFiles):                       
        trip = Trip(driverID,tripID,pd.read_csv(driverDir+'_' + str(tripID) + '.csv'))
        trip.getSpeed()
        trip.getRadius()
        trip.getCacc()
        cacc.loc[index] = asarray(trip.Quantiles(trip.cacc.val)) 
    
    cacc.to_csv('/home/user1/Desktop/Share2Windows/Kaggle/FeaturesCleaned/CaccQuantiles/' + str(driverID) + '.csv', index=False)
    del cacc

    return 0

# <codecell>

import time
from joblib import Parallel, delayed  
import multiprocessing
num_cores=multiprocessing.cpu_count()

'''for driverID in driversFiles:
    print driverID
    GetFeatures(driverID)
'''

#num_cores=2
inputs = range(len(driversFiles)/num_cores)
for i in inputs:
    #start = time.time()
    r=range(num_cores*i,num_cores*i+num_cores)
    #print r
    Parallel(n_jobs=num_cores)(delayed(MakePrediction)(driversFiles[j],j % num_cores) for j in r) 
    #print time.time()-start 

# <codecell>

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
import EnsembleClassifier as eclf
reload(eclf)

randTrips = range(200)
drivers_sampleSize=5
trips_sampleSize=150

def MakePrediction(cur_driver,tk):
    #print cur_driver
    Xpred = pd.read_csv('/home/user1/Desktop/Share2Windows/Kaggle/FeaturesCleaned/CaccQuantiles/' + str(cur_driver) + '.csv')
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
        
        D = pd.read_csv('/home/user1/Desktop/Share2Windows/Kaggle/FeaturesCleaned/CaccQuantiles/' + str(rand_driver) + '.csv')
        np.random.shuffle(randTrips)
        Xtrain = Xtrain.append(D.loc[randTrips[:trips_sampleSize]])
        tmp = pd.DataFrame(np.zeros((trips_sampleSize,1)))
        tmp.ix[:] = int(0)
        ytrain = ytrain.append(tmp)  
        
    Xtrain.index = range(Xtrain.shape[0])
    ytrain.index = range(Xtrain.shape[0])
    
    #preprocessing
    #pca.fit(Xtrain)
    
    
    #fit model  
    #===============================================================================
    clf = eclf.EnsembleClassifier(
    clfs=[
    LR(class_weight='auto',C=0.5)
    ,RF()
    ,GBC()
    ])

    clf.fit(Xtrain,asarray(ytrain[0]))
    #===============================================================================
    cur_driver_df.prob=clf.predict_proba(Xpred)[:,1]
    
    cur_driver_df.to_csv('/home/user1/Desktop/Share2Windows/Kaggle/results/' + str(cur_driver) + '.csv', index=False)  
    
    return 0

# <codecell>

f = lambda x:  '%s_%s' % (int(x[0]),int(x[1]))
main_df = pd.DataFrame(columns=['driver','trip','prob'])
for iter, cur_driver in enumerate(driversFiles):
    cur_driver_df = pd.read_csv('/home/user1/Desktop/Share2Windows/Kaggle/results/' + str(cur_driver) + '.csv')

    if (cur_driver in set(main_df.driver)):
        print 'Error'
        break
        
    main_df=main_df.append(cur_driver_df)
        
main_df.index = range(main_df.shape[0])
main_df['driver_trip'] = main_df.apply(f,axis=1)
main_df[['driver_trip','prob']].to_csv('/home/user1/Desktop/Share2Windows/Kaggle/caccQuantiles.csv',  index=False)

# <codecell>

main_df.head()

# <codecell>


