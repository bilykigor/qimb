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

drivers = map(int,os.listdir('../Kaggle/drivers/'))
drivers.sort(reverse=False)

# <codecell>

drivers[-5:]

# <codecell>

featuresDir = '/home/user1/Desktop/Share2Windows/Kaggle/features/'
drivers = map(int,os.listdir('../Kaggle/drivers/'))
drivers.sort(reverse=False)
vlim=[1,50]
clim=[-4,4]
trips = range(1,201)

n_components = [5]#[2,4,6,8,10]
selectedCols = ['acc','acc2','v']
clusterModel = KMeans(n_clusters=2)

def GetFeatures(driverID, j):
    N=5000
    #print driverID
    driverDir = '../Kaggle/drivers/'+str(driverID)
    
    cur_driver_df = pd.DataFrame(np.zeros((200,3)),columns=['driver','trip','prob'])
    cur_driver_df.driver = driverID
    cur_driver_df.trip = trips    
   
    X = pd.DataFrame(columns=selectedCols)
    for index,tripID in enumerate(trips):                       
        trip = Trip(driverID,tripID,pd.read_csv(driverDir+'/' + str(tripID) + '.csv'))
        trip.features[selectedCols].to_csv(featuresDir+'/' +str(driverID)+'_' + str(tripID) + '.csv', index=False)
        X = X.append(trip.features[selectedCols])
        
    X.index = range(X.shape[0])
    '''X=X[(X.acc2<clim[1]) & (X.acc2>clim[0])]
    X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
    X=X[(X.v<vlim[1]) & (X.v>vlim[0])]
    X.index = range(X.shape[0])'''
    
    x = np.asanyarray(X);np.random.shuffle(x)
    xN = x[:N,:]
    
    #train GMM
    gmms = [GMM(n_components=n, covariance_type='full').fit(xN) for n in n_components]
    BICs = [gmm.bic(xN) for gmm in gmms]
    i_min = np.argmin(BICs)
    clf_driver=gmms[i_min]
    
    distances=np.zeros((200,11))
    for index,tripID in enumerate(trips): 
        N=5000
        #print index
        X = pd.read_csv(featuresDir+'/'+str(driverID)+'_' + str(tripID) + '.csv')
        '''X=X[(X.acc2<clim[1]) & (X.acc2>clim[0])]
        X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
        X=X[(X.v<vlim[1]) & (X.v>vlim[0])]
        X.index = range(X.shape[0])'''
        
        if (X.shape[0]==0):
            continue
           
        N = min(X.shape[0],N)
        x = np.asanyarray(X);np.random.shuffle(x)
        xN = x[:N,:]
        
        #train GMM
        gmms = [GMM(n_components=n, covariance_type='full').fit(xN) for n in n_components]
        BICs = [gmm.bic(xN) for gmm in gmms]
        i_min = np.argmin(BICs)
        clf_trip=gmms[i_min]
        
        #get distance quantiles
        driver_pred = clf_driver.predict(xN)
        trip_pred = clf_trip.predict(xN)
        d=[multivariate_normal.pdf(xN[i,:],
                                 mean=clf_driver.means_[driver_pred[i]],
                                 cov=clf_driver.covars_[driver_pred[i]]) -
           multivariate_normal.pdf(xN[i,:],
                                 mean=clf_trip.means_[trip_pred[i]],
                                 cov=clf_trip.covars_[trip_pred[i]])
         for i in range(xN.shape[0])]
         
        distances[index,:]=np.percentile(d,[x*10 for x in range(11)])
    
    clusters = list(clusterModel.fit_predict(distances))
    c_targer=argmax([clusters.count(k) for k in range(3)])
    cur_driver_df.prob=0
    cur_driver_df.prob.loc[clusters==c_targer]=1
            
    #print '%s badDistances' % (200-cur_driver_df.prob.sum() )
    cur_driver_df.to_csv('/home/user1/Desktop/Share2Windows/Kaggle/results/' + str(driverID) + '.csv', index=False)
        
    return clf_driver

# <codecell>

def GetFeatures(driverID, j):
    driverDir = '../Kaggle/drivers/'+str(driverID)
    
    for index,tripID in enumerate(trips):                       
        trip = Trip(driverID,tripID,pd.read_csv(driverDir+'/' + str(tripID) + '.csv'))
        trip.features[selectedCols].to_csv(featuresDir+'/' +str(driverID)+'_' + str(tripID) + '.csv', index=False)
    return 0

# <codecell>

import time
from joblib import Parallel, delayed  
import multiprocessing
num_cores=multiprocessing.cpu_count()

driversFiles = map(int,os.listdir('../Kaggle/drivers/'))
driversFiles.sort(reverse=False)

'''for driverID in driversFiles:
    print driverID
    GetFeatures(driverID)
'''
num_cores=4
inputs = range(len(driversFiles)/num_cores)
for i in inputs:
    start = time.time()
    r=range(num_cores*i,num_cores*i+num_cores)
    Parallel(n_jobs=num_cores)(delayed(GetFeatures)(driversFiles[j],j % num_cores) for j in r) 
    print time.time()-start 

# <codecell>

driversFiles = map(int,os.listdir('../Kaggle/drivers/'))
len(driversFiles)/4.0*130/60/60

# <codecell>

import time
start = time.time()
clf=GetFeatures(1,0)
print time.time()-start 

# <codecell>

X = pd.read_csv(featuresDir+'/'+str(0)+'_' + str(24) + '.csv')
X=X[(X.acc2<clim[1]) & (X.acc2>clim[0])]
X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
X.index = range(X.shape[0])
xN = np.asanyarray(X)
#train GMM
gmms = [GMM(n_components=n, covariance_type='full').fit(xN) for n in n_components]
BICs = [gmm.bic(xN) for gmm in gmms]
i_min = np.argmin(BICs)
clf_trip=gmms[i_min]

# <codecell>


# <codecell>

start = time.time()
driver_pred = clf.predict(xN)
trip_pred = clf_trip.predict(xN)
distances1=[multivariate_normal.pdf(xN[i,:],
                         mean=clf.means_[driver_pred[i]],
                         cov=clf.covars_[driver_pred[i]]) -
 multivariate_normal.pdf(xN[i,:],
                         mean=clf_trip.means_[trip_pred[i]],
                         cov=clf_trip.covars_[trip_pred[i]])
 for i in range(xN.shape[0])]
print time.time()-start

# <codecell>

plt.plot(np.percentile(distances,[x*10 for x in range(11)]))
plt.plot(np.percentile(distances1,[x*10 for x in range(11)]))

# <codecell>


