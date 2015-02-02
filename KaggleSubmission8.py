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

''' Submission 8'''
from Trip import Trip
from sklearn.mixture import GMM

featuresDir = '/home/user1/Desktop/Share2Windows/Kaggle/features'
tripFiles = range(1,201)

vlim=[0,40]
clim=[-4,4]
n_components = [2,5,10]#np.arange(1,10)

def GetFeatures(driverID, j):
    #print driverID
    driverDir = '../Kaggle/drivers/'+str(driverID)
    
    cur_driver_df = pd.DataFrame(np.zeros((200,4000)))
   
    for index,tripID in enumerate(tripFiles):    
        #print tripID
        trip = Trip(driverID,tripID,pd.read_csv(driverDir+'/' + str(tripID) + '.csv'))
        X = trip.features
        X=X[(X.v<vlim[1]) & (X.v>vlim[0])]
        X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
        X.index = range(X.shape[0])    
        xN = np.asanyarray(X)
    
        #train GMM
        #gmms = [GMM(n_components=n, covariance_type='full').fit(xN) for n in n_components]
        #BICs = [gmm.bic(xN) for gmm in gmms]
        #i_min = np.argmin(BICs)
        #clf=gmms[i_min]
        #print '%s components' %(n_components[i_min])
        try:
            clf = GMM(n_components=5, covariance_type='full').fit(xN)

            X_, Y_ = np.meshgrid(np.linspace(clim[0], clim[1],num=80),np.linspace(vlim[0], vlim[1]),num=40)
            XX = np.array([X_.ravel(), Y_.ravel()]).T
            Z = np.exp(clf.score(XX))
            cur_driver_df.loc[tripID] = Z
        except:
            print 'exception driver %d trip %d' %(driverID,tripID)

    cur_driver_df.loc[1:].to_csv(featuresDir+'/' + str(driverID) + '.csv', index=False)
        
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
    #start = time.time()
    r=range(num_cores*i,num_cores*i+num_cores)
    Parallel(n_jobs=num_cores)(delayed(makePrediction)(driversFiles[j]) for j in r) 
    #print time.time()-start 

# <codecell>

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
drivers = map(int,os.listdir('../Kaggle/drivers/'))
randomDrivers = map(int,os.listdir('../Kaggle/drivers/'))
drivers.sort(reverse=False)
resultsDir = '/home/user1/Desktop/Share2Windows/Kaggle/results'
drivers_sampleSize=10
trips_sampleSize=200
rand_line_inds = range(trips_sampleSize)
main_df = pd.DataFrame(columns=['driver','trip','prob'])
pca = PCA(n_components=500)

def makePrediction(cur_driver):
    #print cur_driver
    Xpred = pd.read_csv(featuresDir+'/' + str(cur_driver) + '.csv')
    cur_driver_df = pd.DataFrame(np.zeros((Xpred.shape[0],3)),columns=['driver','trip','prob'])
    cur_driver_df.driver = cur_driver
    cur_driver_df.trip = range(1,201)
    
    np.random.shuffle(rand_line_inds)
    Xtrain = Xpred.loc[rand_line_inds[:trips_sampleSize]]
    ytrain = pd.DataFrame(np.ones((trips_sampleSize,1)))
    #ytrain *= cur_driver
    
    np.random.shuffle(randomDrivers)           
    for rand_driver in randomDrivers[:drivers_sampleSize]:
        if rand_driver==cur_driver:
            continue
        
        D = pd.read_csv(featuresDir+'/' + str(rand_driver) + '.csv')
        np.random.shuffle(rand_line_inds)
        Xtrain = Xtrain.append(D.loc[rand_line_inds[:trips_sampleSize]])
        tmp = pd.DataFrame(np.zeros((trips_sampleSize,1)))
        #tmp.ix[:] =  rand_driver
        ytrain = ytrain.append(tmp)  
        
    Xtrain.index = range(Xtrain.shape[0])
    ytrain.index = range(Xtrain.shape[0])
    
    #preprocessing
    #pca.fit(Xtrain)
    
    
    #fit model  
    #===============================================================================
    #clf = LR(class_weight='auto',C=0.1)
    #clf = KNN(n_neighbors= 3)
    
    clf = RF()#n_jobs=4
    '''weights = ytrain.copy()
    weights[0][ytrain[0]==1]=ytrain[ytrain[0]==0].shape[0]
    weights[0][ytrain[0]==0]=ytrain[ytrain[0]==1].shape[0]
    clf.fit(Xtrain.ix[:,2:],asarray(ytrain[0]), sample_weight=asarray(weights[0]))'''
    
    #clf = GBC()
    #clf = SVC()
    
    #
    clf.fit(Xtrain,asarray(ytrain[0]))
    
    fi = pd.DataFrame()
    fi['Feature'] = list(Xtrain.columns)
    fi['Impotrance'] = clf.feature_importances_
    fi=fi.sort(columns=['Impotrance'],ascending=False)
    fi['Index'] = range(Xtrain.shape[1])
    fi.index = fi['Index']
    
    pca.fit(Xtrain[fi['Feature'][:1000]])
    
    Xtrain = pca.transform(Xtrain[fi['Feature'][:1000]])
    
    clf = GBC()
    clf.fit(Xtrain,asarray(ytrain[0]))
    #===============================================================================
    Xpred = pca.transform(Xpred[fi['Feature'][:1000]])
    cur_driver_df.prob = 0
    cur_driver_df.prob=clf.predict(Xpred) 
    
    cur_driver_df.to_csv(resultsDir+'/' + str(cur_driver) + '.csv', index=False)  
    
    #print list(cur_driver_df.prob).count(0)
    
    return 0

# <codecell>

c=makePrediction(1)

# <codecell>

main_df = pd.DataFrame(columns=['driver','trip','prob'])
for iter, driverID in enumerate(driversFiles):
    cur_driver_df = pd.read_csv(resultsDir+'/' + str(driverID) + '.csv')

    if (driverID in set(main_df.driver)):
        print 'Error'
        break
        
    main_df=main_df.append(cur_driver_df)
    
    #if iter>1:
    #    break
    
main_df.index = range(main_df.shape[0])
main_df.to_csv('/home/user1/Desktop/Share2Windows/Kaggle/KNNv_acc.csv',  index=False)

# <codecell>


# <codecell>


