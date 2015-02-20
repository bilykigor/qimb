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
from sklearn.neighbors import KernelDensity as kde
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis

# <codecell>

driverID = 1
driverDir = '/home/user1/Desktop/SharedFolder/Kaggle/DriversCleaned/'+str(driverID)
tripInd = 1
df = pd.read_csv(driverDir+'_' + str(tripInd)+'.csv')
trip = Trip(driverID,tripInd,df)
trip.getSpeed()
trip.getAcc()
#trip.getRadius()
#trip.getCacc()
trip.getFeatures()
X=trip.features[['v','acc']]

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

driverID=10
drive=pd.read_csv('/home/user1/Desktop/SharedFolder/Kaggle/FeaturesCleaned/GMM/All/' + str(driverID) + '.csv')

d = pd.DataFrame(np.zeros((200,2)),columns=['val1','val2'])
for i in range(1,201):
    d.loc[i-1]=[GetProba(drive,driverID,i),GetProba(drive,2,i)]

# <codecell>

ggplot(d,aes(x='val1')) + geom_histogram(fill='green',alpha=0.4)+\
geom_histogram(d,aes(x='val2'),fill='red',alpha=0.4)

# <codecell>

z=array(list(set(np.asarray([range(x-10,x+10) for x in (X.v<vlim[0]).nonzero()[0]]).flatten())))
z=z[z<X.shape[0]]
z=z[z>=0]
#z=array(list(set(range(X.shape[0]))-set(z)))

# <codecell>

Xz=X.loc[z]
Xz=Xz.reset_index(drop=True)

# <codecell>

tripFiles = range(1,201)
vlim=[0.01,15]
clim=[-10,10]
selectedCols = ['acc','cacc','v']

# <codecell>

selectedCols = ['v','acc']

# <codecell>

X = pd.DataFrame(columns=selectedCols)
for tripID in range(1,201):                       
    trip = Trip(driverID,tripID,pd.read_csv(driverDir+'_' + str(tripID) + '.csv'))
    trip.getSpeed()
    trip.getAcc()
    #trip.getRadius()
    #trip.getCacc()
    trip.getFeatures()
    '''z=array(list(set(np.asarray([range(x-5,x+5) for x in (trip.features.v<vlim[0]).nonzero()[0]]).flatten())))
    z=z[z<trip.features.shape[0]]
    z=z[z>=0]
    z=array(list(set(range(trip.features.shape[0]))-set(z)))
    
    Xz=trip.features.loc[z]
    Xz=Xz.reset_index(drop=True)
    
    Xz=Xz.loc[Xz.v!=0]
    Xz=Xz.reset_index(drop=True)'''

    X = X.append(trip.features)
X=X.reset_index(drop=True)

# <codecell>

vlim=[0.01,40]
clim=[-20,20]
X=X[(X.v<vlim[1]) & (X.v>vlim[0])]
X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
X=X[(X.acc2<clim[1]) & (X.acc2>clim[0])]
X.index = range(X.shape[0])
X=X[selectedCols]

# <codecell>

from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity

'''grid = GridSearchCV(KernelDensity(),{'bandwidth': np.linspace(0.1, 1.0, 5)},cv=10,n_jobs=2)
grid.fit(np.asanyarray(X[['v','acc']]))
print grid.best_params_'''
clf = KernelDensity()
clf.fit(X[['v','acc']])

# <codecell>

def GetProba(clf,driverID,tripID):
    #print driverID,tripID
    driverDir = '/home/user1/Desktop/SharedFolder/Kaggle/DriversCleaned/'+str(driverID)
    df = pd.read_csv(driverDir+'_' + str(tripInd)+'.csv')
    trip = Trip(driverID,tripInd,df)
    trip.getSpeed()
    trip.getAcc()
    #trip.getRadius()
    #trip.getCacc()
    trip.getFeatures()
    X=trip.features[['v','acc']]
    
    probas = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        probas[i]=clf.score(X.loc[i])
    print probas.mean()
    return probas.mean()

# <codecell>

driverID=1
d = pd.DataFrame(np.zeros((200,2)),columns=['val1','val2'])
for i in range(1,200):
    d.loc[i-1]=[GetProba(clf,driverID,i),GetProba(clf,3,i)]
    print d.loc[i-1]

# <codecell>

driverID=2
tripInd=2
driverDir = '/home/user1/Desktop/SharedFolder/Kaggle/DriversCleaned/'+str(driverID)
df = pd.read_csv(driverDir+'_' + str(tripInd)+'.csv')
trip = Trip(driverID,tripInd,df)
trip.getSpeed()
trip.getAcc()
    #trip.getRadius()
    #trip.getCacc()
trip.getFeatures()
X=trip.features[['v','acc']]
    
probas = np.zeros(X.shape[0])
    
for i in range(X.shape[0]):
    probas[i]=clf.score(X.loc[i])

# <codecell>

probas.mean()

# <codecell>

sns.jointplot(X.v,X.acc,kind = "scatter",size=6,ratio=5,marginal_kws={'bins':30})
#sns.kdeplot(X[['cacc','acc']])

# <codecell>

xN = np.asanyarray(X[['cacc','acc']])

# <codecell>

n_components = range(1,25)
gmms = [GMM(n_components=n, covariance_type='full').fit(xN) for n in n_components]
BICs = [gmm.bic(xN) for gmm in gmms]
i_min = np.argmin(BICs)
clf=gmms[i_min]
print '%s components - BIC %s' %(n_components[i_min],BICs[i_min])
tol = np.percentile(np.exp(clf.score(xN)),10)

# <codecell>

fig1 = plt.figure(figsize=(10, 10))
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm','wheat','violet','sienna',
                              'yellow','springgreen','sandybrown','tomato','tan','teal'])

Y_ = clf.predict(xN)
for i, (mean, covar, color) in enumerate(zip(clf.means_, clf._get_covars(), color_iter)):
    v, w = linalg.eigh(covar)
    u = w[0] / linalg.norm(w[0])

    if not np.any(Y_ == i):
        continue
    plt.scatter(xN[Y_ == i, 0], xN[Y_ == i, 1], .8, color=color)

'''X_, Y_ = np.meshgrid(np.linspace(vlim[0], vlim[1]),np.linspace(clim[0], clim[1]))
XX = np.array([X_.ravel(), Y_.ravel()]).T
Z = np.exp(clf.score(XX))
Z = Z.reshape(X_.shape)
CS = plt.contour(X_, Y_, Z)'''

plt.show()

# <codecell>


# <codecell>

featuresDir = '/home/user1/Desktop/Share2Windows/Kaggle/features/'
drivers = map(int,os.listdir('../Kaggle/drivers/'))
drivers.sort(reverse=False)
vlim=[5,50]
clim=[-4,4]

X = pd.DataFrame(columns=['acc','v'])
for tripID in xrange(1,201):    
    X = X.append(pd.read_csv(featuresDir + str(driverID) +'_'+str(tripID) + '.csv'))
    
X.index = range(X.shape[0])
X=X[(X.v<vlim[1]) & (X.v>vlim[0])]
X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
X.index = range(X.shape[0])

# <codecell>

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.mixture import GMM
from sklearn.neighbors import KernelDensity
np.random.seed(0)
x = np.asanyarray(X[X.v>5].acc)
#------------------------------------------------------------
# plot the results
fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(bottom=0.08, top=0.95, right=0.95, hspace=0.1)
N_values = (10000,20000)
subplots = (211, 212)
np.random.shuffle(x)

for N, subplot in zip(N_values,  subplots):
    ax = fig.add_subplot(subplot)
    xN = x[:N]
   
    # Compute density via Gaussian Mixtures
    # we'll try several numbers of clusters
    n_components = np.arange(2, 10)
    gmms = [GMM(n_components=n, covariance_type='full').fit(xN) for n in n_components]
    BICs = [gmm.bic(xN) for gmm in gmms]
    i_min = np.argmin(BICs)
    t = np.linspace(-10, 30, 1000)
    logprob, responsibilities = gmms[i_min].eval(t)
    
    # plot the results
    ax.plot(t, np.exp(logprob), '-', color='gray',
    label="Mixture Model\n(%i components)" % n_components[i_min])
    
    # label the plot
    ax.text(0.02, 0.95, "%i points" % N, ha='left', va='top',
    transform=ax.transAxes)
    ax.set_ylabel('$p(x)$')
    ax.legend(loc='upper right')
    if subplot == 212:
        ax.set_xlabel('$x$')
plt.show()

# <codecell>

clf.covars_[0]

# <codecell>

clf.covars_[0].flatten()

# <codecell>

start = 0
end = 5

# <codecell>

ggplot(Xz,aes(x='v',y='acc')) + geom_point(size=10)

# <codecell>

df=trip.coordinates.ix[start:end,:].copy()

# <codecell>

i=2
df.x[i]=df.x[i]-df.x[i-1]
df.y[i]=df.y[i]-df.y[i-1]

# <codecell>

df

# <codecell>

ggplot(df,aes(x='x',y='y')) + geom_point(size=5)

# <codecell>

df = trip.cacc.copy()
df['ind'] = range(df.shape[0])
ggplot(df.ix[start:end,:],aes(x='ind',y='val')) + geom_point(size=10)

# <codecell>

df = tripc.features.copy()
df['ind'] = range(df.shape[0])
ggplot(df.ix[start:end,:],aes(x='ind',y='acc')) + geom_point(color='green',size=10)#+\
#geom_point(df.ix[start:end,:],aes(x='ind',y='cacc'),color='red',size=10)

# <codecell>


