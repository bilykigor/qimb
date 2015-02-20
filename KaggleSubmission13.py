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

tripFiles = range(1,201)
driversFiles = map(int,os.listdir('/home/user1/Desktop/SharedFolder/Kaggle/DriversOriginal/'))
randomDrivers = map(int,os.listdir('/home/user1/Desktop/SharedFolder/Kaggle/DriversOriginal/'))
driversFiles.sort(reverse=False)

# <codecell>

cur_driver=3
df=pd.read_csv('/home/user1/Desktop/SharedFolder/Kaggle/results/' + str(cur_driver) + '_1.csv', header=None)

# <codecell>

c=pd.DataFrame(columns=['x','y','v'])
c.x = df.ix[:,:4].values.flatten()
c.y = df.ix[:,5:].values.flatten()
c.v = floor(c.index/5)

# <codecell>

ggplot(c, aes(x='x', y='y', color='v')) +  geom_line() #+ geom_point()

# <codecell>

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
import EnsembleClassifier as eclf
reload(eclf)

randTrips = range(1,200)
drivers_sampleSize=5
trips_sampleSize=200

def MakePrediction(cur_driver,tk):
    #print cur_driver
    cur_driver_df = pd.DataFrame(np.zeros((200,4)),columns=['driver','trip','prob','n'])
    cur_driver_df.driver = cur_driver
    cur_driver_df.trip = range(1,201)
    cur_driver_df.n = 0
    
    Xtrain=pd.read_csv('/home/user1/Desktop/SharedFolder/Kaggle/FeaturesCleaned/' + str(cur_driver) + '_1.csv', header=None)
    cur_driver_df.n[0]=Xtrain.shape[0]
    for i in range(2,201):
        c = pd.read_csv('/home/user1/Desktop/SharedFolder/Kaggle/FeaturesCleaned/' + str(cur_driver) + '_' + str(i) + '.csv', header=None)
        cur_driver_df.n[i-1]=c.shape[0]
        Xtrain=Xtrain.append(c) 
        
    nOne = Xtrain.shape[0]
    
    np.random.shuffle(randomDrivers)           
    for rand_driver in randomDrivers[:drivers_sampleSize]:
        for i in range(1,201):
            Xtrain=Xtrain.append(
            pd.read_csv('/home/user1/Desktop/SharedFolder/Kaggle/FeaturesCleaned/' + str(rand_driver) + '_' + str(i) + '.csv', header=None)) 
 
    ytrain = pd.DataFrame(np.zeros((Xtrain.shape[0],1)))  
    ytrain.values[:nOne] = int(1)
        
    Xtrain=Xtrain.reset_index(drop=True)
    ytrain=ytrain.reset_index(drop=True)
   
    #preprocessing
    #pca.fit(Xtrain)    
    
    #fit model  
    #===============================================================================
    clf = LR(class_weight='auto',C=0.5)
    '''\
    eclf.EnsembleClassifier(
    clfs=[
    LR(class_weight='auto',C=0.5)
    ,RF()
    ,GBC()
    ])'''
    
    clf.fit(Xtrain,ytrain[0])
    #===============================================================================
    start = 0
    finish = 0
    for i in range(1,201):
        finish = start+cur_driver_df.n[i-1]
        cTrip = clf.predict(Xtrain.loc[start:finish-1])
        start=finish
        
        cur_driver_df.prob.values[i-1]=cTrip.sum()/cTrip.shape[0]
    
    cur_driver_df.to_csv('/home/user1/Desktop/SharedFolder/Kaggle/results/' + str(cur_driver) + '.csv', index=False) 
    
    del cur_driver_df
    
    return 0

# <codecell>

ct=MakePrediction(1,1)

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
inputs = range(len(driversFiles)/num_cores)
for i in inputs:
    start = time.time()
    r=range(num_cores*i,num_cores*i+num_cores)
    Parallel(n_jobs=num_cores)(delayed(MakePrediction)(driversFiles[j],j % num_cores) for j in r) 
    #print time.time()-start 

# <codecell>

f = lambda x:  '%s_%s' % (int(x[0]),int(x[1]))
main_df = pd.DataFrame(columns=['driver','trip','prob','n'])
for iter, cur_driver in enumerate(driversFiles):
    cur_driver_df = pd.read_csv('/home/user1/Desktop/SharedFolder/Kaggle/results/' + str(cur_driver) + '.csv')

    if (cur_driver in set(main_df.driver)):
        print 'Error'
        break
        
    main_df=main_df.append(cur_driver_df)
        
main_df.index = range(main_df.shape[0])
main_df['driver_trip'] = main_df.apply(f,axis=1)
main_df[['driver_trip','prob']].to_csv('/home/user1/Desktop/SharedFolder/Kaggle/lr.csv',  index=False)

# <codecell>

main_df[['driver_trip','prob']].to_csv('/home/user1/Desktop/SharedFolder/Kaggle/lr2.csv',  index=False)

# <codecell>

main_df.head()

# <codecell>


