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

allData = pd.read_csv(speedDir+'/' + str(driversFiles[0]) + '.csv')
for driver in driversFiles[1:]:
    df = pd.read_csv(speedDir+'/' + str(driver) + '.csv')
    allData = allData.append(df)
    #print driver
allData.index = range(allData.shape[0])

data = allData.apply(norm,axis=1)

X = data.ix[:,3:]
y = data.ix[:,1]

# <codecell>

from sklearn.ensemble import RandomForestClassifier as RF
clf = RF(criterion = 'entropy')
clf.fit(X,y)

