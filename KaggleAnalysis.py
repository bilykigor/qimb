# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import numpy as np
import pandas as pd
from ggplot import *
from sklearn.preprocessing import PolynomialFeatures
import os
from Trip import Trip

# <codecell>

trips = []

driverID = 1
driverDir = '../Kaggle/drivers/'+str(driverID)
tripFiles = os.listdir(driverDir)

for index, tripFile in enumerate(tripFiles):
    if (tripFile.split('.')[0][-1]!='f'):
        df = pd.read_csv(driverDir+'/' + tripFile)
        trip = Trip(driverID,int(tripFile.split('.')[0]),df)
        trips.append(trip)  

        if index>2:
            break

# <codecell>

trip.features.to_csv('1.csv', index=False)

# <codecell>

tripInd = 1
start = 0
end = trips[tripInd].n

# <codecell>

trips[tripInd].Radius(46)
#Radius(46)

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
ggplot(df.ix[start:end,:],aes(x='ind',y='acc')) + geom_point(color='green',size=10)#+\
#geom_point(df.ix[start:end,:],aes(x='ind',y='cacc'),color='red',size=10)

# <codecell>

ggplot(trips[tripInd].features[trips[tripInd].features.v>=0].ix[start:end,:],aes(x='acc',y='v')) + geom_point(size=10)

# <codecell>

ggplot(trips[tripInd].features[trips[tripInd].features.v>=5].ix[start:end,:],aes(x='acc')) + geom_histogram(binwidth = 0.1)

# <codecell>

ggplot(trips[tripInd].features[trips[tripInd].features.v>=0].ix[start:end,:],aes(x='v')) + geom_histogram(binwidth = 0.5)

# <codecell>

(n,b,p) = matplotlib.pyplot.hist(list(trips[tripInd].features[trips[tripInd].features.v>=0].acc),bins=50,range=[-5,5])

# <codecell>

(n,b,p) = matplotlib.pyplot.hist(list(trips[tripInd].features[trips[tripInd].features.v>=0].v),bins=50,range=[0,40])

# <codecell>

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture

# <codecell>

gmm = mixture.GMM(n_components=5, covariance_type='full')
gmm.fit(trips[tripInd].features)

# <codecell>

dpgmm = mixture.DPGMM(n_components=5, covariance_type='full')
dpgmm.fit(trips[tripInd].features)

# <codecell>

fig1 = plt.figure(figsize=(15, 10))
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
for i, (clf, title) in enumerate([(gmm, 'GMM'),(dpgmm, 'Dirichlet Process GMM')]):
#enumerate([(dpgmm, 'Dirichlet Process GMM')]):
    splot = plt.subplot(2, 2, 1 + i)
    Y_ = clf.predict(trips[tripInd].features)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(trips[tripInd].features.ix[Y_ == i, 0], trips[tripInd].features.ix[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)

plt.show()

# <codecell>

gmm.bic(trips[tripInd].features)

# <codecell>

dpgmm.bic(trips[tripInd].features)

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
n_samples = 300
# generate random sample, two components
np.random.seed(0)
# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
# fit a Gaussian Mixture Model with two components
clf = mixture.GMM(n_components=2, covariance_type='full')
clf.fit(X_train)
# display predicted scores by the model as a contour plot
x = np.linspace(-20.0, 30.0)
y = np.linspace(-20.0, 40.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)[0]
Z = Z.reshape(X.shape)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)
plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()

# <codecell>


