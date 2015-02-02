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

# <codecell>

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

def GetFeatures(driverID):
    startTime=time.time()
    F = pd.DataFrame(np.zeros((1,52)))
    uniqueInd=0
    driverDir = '../Kaggle/drivers/'+str(driverID)
    tripFiles = os.listdir(driverDir)
    #print driverID
    for index, tripFile in enumerate(tripFiles):
        if (tripFile.split('.')[0][-1]!='f'):
            rawCoords = pd.read_csv(driverDir+'/' + str(tripFile))
            trip = Trip(driverID,driverDir+'/' + str(tripFile),rawCoords)
            #(n,b,p) = matplotlib.pyplot.hist(list(trip.features.v),bins=50,range=[0,40])
            #(n,b,p) = matplotlib.pyplot.hist(list(trip.features.cacc),bins=50,range=[0,5])
            (n,b,p) = matplotlib.pyplot.hist(list(trip.features.acc),bins=50,range=[-5,5])
            F.ix[uniqueInd,2:]=n
            F.ix[uniqueInd,0]=driverID
            F.ix[uniqueInd,1]=trip.ID
            
            #print trip.ID
            uniqueInd+=1
    F=F.sort(columns=[1])
    F.index = range(F.shape[0])
    F.to_csv('../Kaggle/driversFeatures/' + str(driverID) + '.csv')
    del F
    return 0
    #print time.time()-startTime

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
num_cores=2
inputs = range(len(driversFiles)/num_cores)
for i in inputs:
    r=range(num_cores*i,num_cores*i+num_cores)
    Parallel(n_jobs=num_cores)(delayed(GetFeatures)(driversFiles[j]) for j in r) 

# <codecell>

from Trip import Trip
def GetFeatures(driverID):
    driverDir = '../Kaggle/drivers/'+str(driverID)
    tripFiles = range(1,201)
   
    F = pd.DataFrame(np.zeros((1,20)))
    for index,tripID in enumerate(tripFiles):                       
        trip = Trip(driverID,tripID,pd.read_csv(driverDir+'/' + str(tripID) + '.csv'))
        F.ix[index,0] = tripID
        F.ix[index,1:] = asarray(trip.accQuantiles)
    
    F.to_csv('../Kaggle/driversFeatures/' + str(driverID) + '.csv')
    del F
    return 0

# <codecell>

''' Submission 6'''
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestClassifier as RF
drivers = map(int,os.listdir('../Kaggle/drivers/'))
drivers.sort(reverse=False)
randomDrivers = asarray(drivers)
#featuresDir = '../Kaggle/features/speedQuantiles'
featuresDir = '../Kaggle/features/accQuantiles'
drivers_sampleSize=5
trips_sampleSize=200

main_df = pd.DataFrame(columns=['driver','trip','prob'])
rand_line_inds = range(200)
for iter, cur_driver in enumerate(drivers):
    #print cur_driver
    Xpred = pd.read_csv(featuresDir+'/' + str(cur_driver) + '.csv')
    cur_driver_df = pd.DataFrame(np.zeros((Xpred.shape[0],3)),columns=['driver','trip','prob'])
    cur_driver_df.driver = cur_driver
    cur_driver_df.trip = Xpred.ix[:,1]
    
    np.random.shuffle(rand_line_inds)
    Xtrain = Xpred.loc[rand_line_inds[:trips_sampleSize]]
    ytrain = pd.DataFrame(np.ones((trips_sampleSize,1)))
    
    np.random.shuffle(randomDrivers)           
    for rand_driver in randomDrivers[:drivers_sampleSize]:
        if rand_driver==cur_driver:
            continue
        
        D = pd.read_csv(featuresDir+'/' + str(rand_driver) + '.csv')
        np.random.shuffle(rand_line_inds)
        Xtrain = Xtrain.append(D.loc[rand_line_inds[:trips_sampleSize]])
        ytrain = ytrain.append(pd.DataFrame(np.zeros((trips_sampleSize,1))))    

    #clf = LR(class_weight='auto',C=0.1)
    
    '''clf = RF(n_jobs=4)
    weights = ytrain.copy()
    weights[0][ytrain[0]==1]=ytrain[ytrain[0]==0].shape[0]
    weights[0][ytrain[0]==0]=ytrain[ytrain[0]==1].shape[0]
    clf.fit(Xtrain.ix[:,2:],asarray(ytrain[0]), sample_weight=asarray(weights[0]))
    '''
    clf = GBC()
    #clf = SVC(class_weight='auto',probability=True,kernel='sigmoid',C=0.1)

    clf.fit(Xtrain.ix[:,2:],asarray(ytrain[0]))

    cur_driver_df.prob = clf.predict_proba(Xpred.ix[:,2:])[:,1]
    if (cur_driver in set(main_df.driver)):
        print 'Error'
        break
        
    main_df=main_df.append(cur_driver_df)
    
    #if iter>1:
    #    break
    
main_df.index = range(main_df.shape[0])
main_df.to_csv('../Kaggle/GBCaccQuantiles.csv',  index_col=None)

# <codecell>

''' Submission 7'''
from Trip import Trip
from sklearn.mixture import GMM

featuresDir = '../Kaggle/features'
tripFiles = range(1,201)

vlim=[5,50]
clim=[-5,5]
N=10000
Percentile = 10
n_components = np.arange(1, 20)

def GetFeatures(driverID, j):
    #print driverID
    driverDir = '../Kaggle/drivers/'+str(driverID)
    
    cur_driver_df = pd.DataFrame(np.zeros((200,3)),columns=['driver','trip','prob'])
    cur_driver_df.driver = driverID
    cur_driver_df.trip = tripFiles    
   
    X = pd.DataFrame(columns=['acc','v'])
    for index,tripID in enumerate(tripFiles):                       
        trip = Trip(driverID,tripID,pd.read_csv(driverDir+'/' + str(tripID) + '.csv'))
        trip.features.to_csv(featuresDir+'/' +str(j)+'_' + str(tripID) + '.csv', index=False)
        X = X.append(trip.features)
        
    X.index = range(X.shape[0])
    X=X[(X.v<vlim[1]) & (X.v>vlim[0])]
    X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
    X.index = range(X.shape[0])
    
    x = np.asanyarray(X);np.random.shuffle(x)
    xN = x[:N,:]
    
    #train GMM
    gmms = [GMM(n_components=n, covariance_type='full').fit(xN) for n in n_components]
    BICs = [gmm.bic(xN) for gmm in gmms]
    i_min = np.argmin(BICs)
    clf=gmms[i_min]
    #print '%s components' %(n_components[i_min])
    tol = np.percentile(np.exp(clf.score(xN)),Percentile)

    distances=np.zeros(200)
    for index,tripID in enumerate(tripFiles):    
        X = pd.read_csv(featuresDir+'/'+str(j)+'_' + str(tripID) + '.csv')
        X=X[(X.v<vlim[1]) & (X.v>vlim[0])]
        X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
        X.index = range(X.shape[0])
        
        if (X.shape[0]==0):
            continue
            
        x = np.asanyarray(X);np.random.shuffle(x)
        xN = x[:N,:]
        distances[tripID-1] = np.median(np.exp(clf.score(xN)))
        if distances[tripID-1]>tol:
            cur_driver_df.prob[tripID-1]=1
        
    badDistances = [i+1 for i,x in enumerate(distances) if x<tol]
    #print '%s badDistances' % (len(badDistances)   )
    cur_driver_df.to_csv(featuresDir+'/results/' + str(driverID) + '.csv', index=False)
        
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
    Parallel(n_jobs=num_cores)(delayed(GetFeatures)(driversFiles[j],j % num_cores) for j in r) 
    #print time.time()-start 

# <codecell>

''' Submission 7'''
main_df = pd.DataFrame(columns=['driver','trip','prob'])
for iter, driverID in enumerate(driversFiles):
    cur_driver_df = pd.read_csv(featuresDir+'/results/' + str(driverID) + '.csv')

    if (driverID in set(main_df.driver)):
        print 'Error'
        break
        
    main_df=main_df.append(cur_driver_df)
    
    #if iter>1:
    #    break
    
main_df.index = range(main_df.shape[0])
main_df.to_csv('../Kaggle/GMM.csv',  index=False)

# <codecell>

''' Submission 8'''
from Trip import Trip
from sklearn.mixture import GMM

featuresDir = '../Kaggle/features'
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
        clf = GMM(n_components=5, covariance_type='full').fit(xN)
        
        X_, Y_ = np.meshgrid(np.linspace(clim[0], clim[1],num=80),np.linspace(vlim[0], vlim[1]),num=40)
        XX = np.array([X_.ravel(), Y_.ravel()]).T
        Z = np.exp(clf.score(XX))
        cur_driver_df.loc[tripID] = Z

    cur_driver_df.loc[1:].to_csv(featuresDir+'/results/' + str(driverID) + '.csv', index=False)
        
    return 0

# <codecell>

import time
start = time.time()
Z=GetFeatures(1,0)
print time.time()-start

# <codecell>

Z.head()

