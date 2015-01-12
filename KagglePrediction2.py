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

# <codecell>

#One guy code

speedDistribution <- function(trip)
{
    window = 20
  vitesse = 3.6*sqrt(diff(trip$x,window,1)^2 + diff(trip$y,window,1)^2)/window
  return(quantile(vitesse, seq(0.05,1, by = 0.05)))
}

drivers = list.files("kaggle/Axa/drivers")
randomDrivers = sample(drivers, size = 5)

refData = NULL
target = 0
names(target) = "target"
for(driver in randomDrivers)
{
  dirPath = paste0("kaggle/Axa/drivers/", driver, '/')
  for(i in 1:200)
  {
    trip = read.csv(paste0(dirPath, i, ".csv"))
    features = c(speedDistribution(trip), target)
    refData = rbind(refData, features)
  }
}

target = 1
names(target) = "target"
submission = NULL
for(driver in drivers)
{
  print(driver)
  dirPath = paste0("kaggle/Axa/drivers/", driver, '/')
  currentData = NULL
  for(i in 1:200)
  {
    trip = read.csv(paste0(dirPath, i, ".csv"))
    features = c(speedDistribution(trip), target)
    currentData = rbind(currentData, features)
  }
  train = rbind(currentData, refData)
  train = as.data.frame(train)
  g = glm(target ~ ., data=train, family = binomial("logit"))
  currentData = as.data.frame(currentData)
  p =predict(g, currentData, type = "response")
  labels = sapply(1:200, function(x) paste0(driver,'_', x))
  result = cbind(labels, p)
  submission = rbind(submission, result)
}

colnames(submission) = c("driver_trip","prob")
write.csv(submission, "kaggle/Axa/submission.csv", row.names=F, quote=F)

