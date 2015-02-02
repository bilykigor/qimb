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

# <codecell>

driverID = 10
driverDir = '../Kaggle/drivers/'+str(driverID)
tripInd = 1
df = pd.read_csv(driverDir+'/' + str(tripInd)+'.csv')
trip = Trip(driverID,tripInd,df)
trip.getSpeed()
trip.getAcc()
trip.getFeatures()
X=trip.features

# <codecell>

trip.getRadius()
trip.getCacc()

# <codecell>

tripFiles = range(1,201)
vlim=[5,50]
clim=[-4,4]
selectedCols = ['acc','acc2']

# <codecell>

X = pd.DataFrame(columns=selectedCols)
for index,tripID in enumerate(tripFiles):                       
    trip = Trip(driverID,tripID,pd.read_csv(driverDir+'/' + str(tripID) + '.csv'))
    X = X.append(trip.features)

# <codecell>

#cleaned data
driverID = 10
tripInd = 1
df = pd.read_csv('../Kaggle/10_1.csv')
tripc = Trip(driverID,tripInd,df)
Xc=tripc.features

# <codecell>

X.index = range(X.shape[0])
X=X[(X.v<vlim[1]) & (X.v>vlim[0])]
X=X[(X.acc<clim[1]) & (X.acc>clim[0])]
X=X[(X.acc2<clim[1]) & (X.acc2>clim[0])]
X.index = range(X.shape[0])
X=X[selectedCols]

# <codecell>

sns.jointplot(X.cacc,X.acc,kind = "scatter",size=6,ratio=5,marginal_kws={'bins':30})

# <codecell>

sns.kdeplot(Xc[['acc2','acc']])

# <codecell>

xN = np.asanyarray(X[['acc2','acc']])

# <codecell>

n_components = np.arange(2, 15)
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

X_, Y_ = np.meshgrid(np.linspace(clim[0], clim[1]),np.linspace(clim[0], clim[1]))
XX = np.array([X_.ravel(), Y_.ravel()]).T
Z = np.exp(clf.score(XX))
Z = Z.reshape(X_.shape)
CS = plt.contour(X_, Y_, Z)

plt.show()

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


# <codecell>

start = 0
end = trip.n

# <codecell>

ggplot(trip.coordinates.ix[start:end,:],aes(x='x',y='y')) + geom_point(size=5)

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


