# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import qimbs 
import numpy as np
import pandas as pd
from ggplot import *
from sklearn.preprocessing import PolynomialFeatures
import os

# <codecell>

class Trip:
    driverID = -1
    ID = -1
    coordinates=None
    speed = None
    features = None
    
    fullPath=''
    tripLen = 0
    tripDuration = 0
    
    maxAcc = 36
    maxCentracc = 50
    
    n=0
    
    def __init__(self,dID,fName,df):
        self.driverID = dID
        self.fullPath = fName
        self.coordinates = df
        self.n = df.shape[0]
        self.ID = int(fName.split('/')[-1].split('.')[0])
        self.RemoveDirectionOutliers()
        self.getFeatures()
        self.RemoveSpeedOutliers()
        
    def getTripLen(self):
        self.n = self.coordinates.shape[0]
        self.tripDuration = (self.n-1)/60
        l = 0.0
        for i in range(1,self.n):
            l+=self.Distance(self.coordinates.x[i],self.coordinates.y[i],self.coordinates.x[i-1],self.coordinates.y[i-1])
        self.tripLen = l
    
    def getFeatures(self):
        self.n = self.coordinates.shape[0]
        
        self.speed = pd.DataFrame(np.zeros((self.n-1,1)))
        self.speed.columns = ['m_s']
        
        self.radius = pd.DataFrame(np.zeros((self.n-2,1)))
        self.radius.columns = ['m']
        
        self.features = pd.DataFrame(np.zeros((self.n-2,3)))
        self.features.columns = ['acc','cacc','v']
        
        i=0
        self.speed.m_s[i] = self.Distance(\
                                self.coordinates.x[i+1],self.coordinates.y[i+1],\
                                self.coordinates.x[i],self.coordinates.y[i])
        for i in range(1,self.n-1):
            self.speed.m_s[i] = self.Distance(\
                                self.coordinates.x[i+1],self.coordinates.y[i+1],\
                                self.coordinates.x[i],self.coordinates.y[i])
            
            r = self.Radius(i-1)
            self.radius.m[i-1] = r
            
            self.features.v[i-1]=0.5*(self.speed.m_s[i-1]+self.speed.m_s[i])
            
            if r>1:
                self.features.cacc[i-1] = pow(self.features.v[i-1],2)/r

            self.features.acc[i-1] = self.speed.m_s[i]-self.speed.m_s[i-1]
    
    def Distance(self,x1,y1,x2,y2):
        return pow(pow(x1-x2,2)+pow(y1-y2,2),0.5)
   
    def Radius(self,i):
        a = self.Distance(self.coordinates.x[i],self.coordinates.y[i],self.coordinates.x[i+1],self.coordinates.y[i+1])
        b = self.Distance(self.coordinates.x[i+2],self.coordinates.y[i+2],self.coordinates.x[i+1],self.coordinates.y[i+1])
        c = self.Distance(self.coordinates.x[i],self.coordinates.y[i],self.coordinates.x[i+2],self.coordinates.y[i+2])
        p = 0.5*(a+b+c)
        return 0.25*a*b*c/pow(p*(p-a)*(p-b)*(p-c),0.5)
    
    def Turn(self,w,v):
        norm = numpy.linalg.norm(v)
        m= matrix(zeros((2,2)))
        m[0,0] = v.x/norm;m[0,1] = -v.y/norm
        m[1,0] = v.y/norm;m[1,1] = v.x/norm
        [xn,yn] = m.T*matrix(w).T
        return xn,yn
    
    def RemoveDirectionOutliers(self):
        self.n = self.coordinates.shape[0]
        iteration = 0;
        while True:
            iteration+=1
            errors = 0
            for i in range(self.n-3):
                sample = self.coordinates.ix[i:i+3,:].copy()
                sample.index = range(4)

                #centralize
                sample.x[1]-=sample.x[0];sample.y[1]-=sample.y[0]
                sample.x[2]-=sample.x[0];sample.y[2]-=sample.y[0]
                sample.x[3]-=sample.x[0];sample.y[3]-=sample.y[0]
                sample.x[0]=0;sample.y[0]=0

                sample.x[3],sample.y[3] =  self.Turn(sample.ix[3,:],sample.ix[1,:])
                sample.x[2],sample.y[2] =  self.Turn(sample.ix[2,:],sample.ix[1,:])
                sample.x[1],sample.y[1] =  self.Turn(sample.ix[1,:],sample.ix[1,:])

                if (((sample.y[2]>0) & (sample.y[3]<0)) | ((sample.y[2]<0) & (sample.y[3]>0))):
                    self.coordinates.x[i+2] = 0.5*(self.coordinates.x[i+3]+self.coordinates.x[i+1])
                    self.coordinates.y[i+2] = 0.5*(self.coordinates.y[i+3]+self.coordinates.y[i+1])
                    errors+=1

            if ((errors==0) | (iteration>10)):
                break
    
    def RemoveSpeedOutliers(self):
        n = self.n-1
        iteration = 0;
        move = 0.1
        while True:
            iteration+=1
            errors = 0
            for i in range(1,n-1):
                cond1 = (self.speed.m_s[i]>self.speed.m_s[i-1] + 2.0/3.6) & \
                        (self.speed.m_s[i]>=self.speed.m_s[i+1] - 1.0/3.6)
                cond2 = (self.speed.m_s[i]<self.speed.m_s[i-1] - 2.0/3.6) & \
                        (self.speed.m_s[i]<=self.speed.m_s[i+1] + 1.0/3.6)
                cond3 = (self.speed.m_s[i]>=self.speed.m_s[i-1] - 1.0/3.6) & \
                        (self.speed.m_s[i]>self.speed.m_s[i+1] + 2.0/3.6)
                cond4 = (self.speed.m_s[i]<=self.speed.m_s[i-1] + 1.0/3.6) & \
                        (self.speed.m_s[i]<self.speed.m_s[i+1] - 2.0/3.6)
                    
                if (cond1 | cond3):
                    v = [self.coordinates.x[i] - self.coordinates.x[i+1],\
                         self.coordinates.y[i] - self.coordinates.y[i+1]]
                    norm = numpy.linalg.norm(v)
                    if norm==0:
                        continue
                    v[0]/=norm
                    v[1]/=norm
                    
                    self.coordinates.x[i+1] += move*v[0]
                    self.coordinates.y[i+1] += move*v[1]
                    
                    self.speed.m_s[i] = self.Distance(self.coordinates.x[i+1],self.coordinates.y[i+1],self.coordinates.x[i],self.coordinates.y[i])
                    self.speed.m_s[i+1] = self.Distance(self.coordinates.x[i+1],self.coordinates.y[i+1],self.coordinates.x[i+2],self.coordinates.y[i+2])
                    
                    errors+=1
                    
                if (cond2 | cond4):
                    v = [self.coordinates.x[i-1] - self.coordinates.x[i],\
                         self.coordinates.y[i-1] - self.coordinates.y[i]]
                    norm = numpy.linalg.norm(v)
                    if norm==0:
                        continue
                    v[0]/=norm
                    v[1]/=norm
                    
                    self.coordinates.x[i] += move*v[0]
                    self.coordinates.y[i] += move*v[1]
                                   
                    self.speed.m_s[i-1] = self.Distance(self.coordinates.x[i],self.coordinates.y[i],self.coordinates.x[i-1],self.coordinates.y[i-1])
                    self.speed.m_s[i] = self.Distance(self.coordinates.x[i+1],self.coordinates.y[i+1],self.coordinates.x[i],self.coordinates.y[i])
                    
                    errors+=1
                    
                if (cond2 | cond4 | cond1 | cond3):     
                    for j in range(max(1,i-3),min(i+3,n-1)):
                        #print j
                        r = self.Radius(j-1)
                        self.radius.m[j-1] = r

                        self.features.v[j-1]=0.5*(self.speed.m_s[j-1]+self.speed.m_s[j])

                        if r>1:
                            self.features.cacc[j-1] = pow(self.features.v[j-1],2)/r
                        else:
                            self.features.cacc[j-1] = 0

                        self.features.acc[j-1] = self.speed.m_s[j]-self.speed.m_s[j-1]

            if ((errors==0) | (iteration>10)):
                break
    

# <codecell>

trips = []

driverID = 1
driverDir = '../Kaggle/drivers/'+str(driverID)
tripFiles = os.listdir(driverDir)

for index, tripFile in enumerate(tripFiles):
    df = pd.read_csv(driverDir+'/' + tripFile)
    trips.append(Trip(1,driverDir+'/' + tripFile,df))  

    if index>2:
        break

# <codecell>

tripInd = 1
start = 110
end = 150#trips[tripInd].n

# <codecell>

trips[tripInd].coordinates.ix[start:end,:]

# <codecell>

trips[tripInd].radius.ix[start:end,:]

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
ggplot(df.ix[start:end,:],aes(x='ind',y='acc')) + geom_point(color='green',size=10)+\
geom_point(df.ix[start:end,:],aes(x='ind',y='cacc'),color='red',size=10)

# <codecell>

ggplot(trips[tripInd].features.ix[start:end,:],aes(x='acc',y='cacc')) + geom_point(size=10)

# <codecell>


