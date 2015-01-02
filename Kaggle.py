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
    fullPath=''
    speed = None
    acceleration = None
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
        self.getSpeed()
        #self.RemoveSpeedOutliers()
        self.getAcceleration()
        #self.getCenrtAcceleration()
        
    def getTripLen(self):
        n = self.coordinates.shape[0]
        self.tripDuration = (n-1)/60
        l = 0.0
        for i in range(1,n):
            l+=self.Distance(self.coordinates.x[i],self.coordinates.y[i],self.coordinates.x[i-1],self.coordinates.y[i-1])
        self.tripLen = l
    
    def getSpeed(self):
        n = self.coordinates.shape[0]
        l = 0.0
        self.speed = pd.DataFrame(np.zeros((n-1,2)))
        self.speed.columns = ['ind','km_h']
        self.speed.ind = range(n-1)
        for i in range(n-1):
            self.speed.km_h[i] = 3.6*self.Distance(self.coordinates.x[i+1],self.coordinates.y[i+1],\
                                                   self.coordinates.x[i],self.coordinates.y[i])
    
    def getCenrtAcceleration(self):
        n = self.coordinates.shape[0]
        self.centracceleration = pd.DataFrame(np.zeros((n-2,2)))
        self.centracceleration.columns = ['ind','acc']
        self.centracceleration.ind = range(n-2)
        for i in range(n-2):
            r = self.R(self.coordinates,i)
            self.centracceleration.acc[i] = pow(0.5*(self.speed.km_h[i]/3.6+self.speed.km_h[i+1])/3.6,2)/r
            self.centracceleration.acc[i] /=9.8
    
    def getAcceleration(self):
        n = self.speed.shape[0]
        l = 0.0
        self.acceleration = pd.DataFrame(np.zeros((n-1,3)))
        self.acceleration.columns = ['acc','cacc','v']
        #self.acceleration.ind = range(n-1)
        for i in range(n-1):
            r = self.R(self.coordinates,i)
            
            self.acceleration.v[i]=0.5*(self.speed.km_h[i]/3.6+self.speed.km_h[i+1])/3.6
            
            self.acceleration.cacc[i] = pow(self.acceleration.v[i],2)/r
            #self.centracceleration.cacc[i] /=9.8
            
            self.acceleration.acc[i] = self.speed.km_h[i+1]/3.6-self.speed.km_h[i]/3.6
            #self.acceleration.acc[i] /=9.8
        
    def SmoothCoordinates(self,df):
        return df
        n = df.shape[0]
        newDf = pd.DataFrame(np.zeros((n-4,2)))
        newDf.columns = df.columns
        for i in range(2,n-2):
            newDf.x[i-2] = (df.x[i-2] + df.x[i-1] + df.x[i] + df.x[i + 1]+ df.x[i + 2])/5.0
            newDf.y[i-2] = (df.y[i-2] + df.y[i-1] + df.y[i] + df.y[i + 1]+ df.y[i + 2])/5.0
        return newDf
    
    def Distance(self,x1,y1,x2,y2):
        return pow(pow(x1-x2,2)+pow(y1-y2,2),0.5)

    def Distance2(self,x1,x2):
        return abs(x1-x2)
    
    def R(self,df,i):
        a = self.Distance(df.x[i],df.y[i],df.x[i+1],df.y[i+1])
        b = self.Distance(df.x[i+2],df.y[i+2],df.x[i+1],df.y[i+1])
        c = self.Distance(df.x[i],df.y[i],df.x[i+2],df.y[i+2])
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
        n = self.coordinates.shape[0]
        iteration = 0;
        while True:
            iteration+=1
            errors = 0
            for i in range(n-3):
                sample = self.coordinates.ix[i:i+3,:].copy()
                sample.index = range(4)

                sample.x[1]-=sample.x[0];sample.y[1]-=sample.y[0]
                sample.x[2]-=sample.x[0];sample.y[2]-=sample.y[0]
                sample.x[3]-=sample.x[0];sample.y[3]-=sample.y[0]
                sample.x[0]=0;sample.y[0]=0

                sample.x[3],sample.y[3] = Turn(sample.ix[3,:],sample.ix[1,:])
                sample.x[2],sample.y[2] = Turn(sample.ix[2,:],sample.ix[1,:])
                sample.x[1],sample.y[1] = Turn(sample.ix[1,:],sample.ix[1,:])

                if (((sample.y[2]>0) & (sample.y[3]<0)) | ((sample.y[2]<0) & (sample.y[3]>0))):
                    self.coordinates.x[i+2] = 0.5*(self.coordinates.x[i+3]+self.coordinates.x[i+1])
                    self.coordinates.y[i+2] = 0.5*(self.coordinates.y[i+3]+self.coordinates.y[i+1])
                    errors+=1

            if ((errors==0) | (iteration>10)):
                break
    
    def RemoveSpeedOutliers(self):
        n = self.speed.shape[0]
        iteration = 0;
        move = 0.1
        while True:
            iteration+=1
            errors = 0
            for i in range(1,n-1):
                cond1 = (self.speed.km_h[i]>self.speed.km_h[i-1] + 2) & \
                        (self.speed.km_h[i]>=self.speed.km_h[i+1] - 1)
                cond2 = (self.speed.km_h[i]<self.speed.km_h[i-1] - 2) & \
                        (self.speed.km_h[i]<=self.speed.km_h[i+1] + 1)
                cond3 = (self.speed.km_h[i]>=self.speed.km_h[i-1] - 1) & \
                        (self.speed.km_h[i]>self.speed.km_h[i+1]+2)
                cond4 = (self.speed.km_h[i]<=self.speed.km_h[i-1] + 1) & \
                        (self.speed.km_h[i]<self.speed.km_h[i+1]-2)
                    
                if (cond1 | cond3):
                    v = [self.coordinates.x[i] - self.coordinates.x[i+1],\
                         self.coordinates.y[i] - self.coordinates.y[i+1]]
                    norm = numpy.linalg.norm(v)
                    v[0]/=norm
                    v[1]/=norm
                    
                    self.coordinates.x[i+1] += move*v[0]
                    self.coordinates.y[i+1] += move*v[1]
                    
                    self.speed.km_h[i] = 3.6*self.Distance(self.coordinates.x[i+1],self.coordinates.y[i+1],self.coordinates.x[i],self.coordinates.y[i])
                    self.speed.km_h[i+1] = 3.6*self.Distance(self.coordinates.x[i+1],self.coordinates.y[i+1],self.coordinates.x[i+2],self.coordinates.y[i+2])
                    
                    errors+=1
                    
                if (cond2 | cond4):
                    v = [self.coordinates.x[i-1] - self.coordinates.x[i],\
                         self.coordinates.y[i-1] - self.coordinates.y[i]]
                    norm = numpy.linalg.norm(v)
                    v[0]/=norm
                    v[1]/=norm
                    
                    self.coordinates.x[i] += move*v[0]
                    self.coordinates.y[i] += move*v[1]
                                   
                    self.speed.km_h[i-1] = 3.6*self.Distance(self.coordinates.x[i],self.coordinates.y[i],self.coordinates.x[i-1],self.coordinates.y[i-1])
                    self.speed.km_h[i] = 3.6*self.Distance(self.coordinates.x[i+1],self.coordinates.y[i+1],self.coordinates.x[i],self.coordinates.y[i])
                    
                    errors+=1

            if ((errors==0) | (iteration>10)):
                break
    
    def RemoveSpeedOutliers2(self):
        n = self.speed.shape[0]
        iteration = 0;
        while True:
            iteration+=1
            errors = 0
            for i in range(1,n-1):
                cond1 = (self.speed.km_h[i]>self.speed.km_h[i-1] + 2) & \
                        (self.speed.km_h[i]>=self.speed.km_h[i+1] - 1)
                cond2 = (self.speed.km_h[i]<self.speed.km_h[i-1] - 2) & \
                        (self.speed.km_h[i]<=self.speed.km_h[i+1] + 1)
                cond3 = (self.speed.km_h[i]>=self.speed.km_h[i-1] - 1) & \
                        (self.speed.km_h[i]>self.speed.km_h[i+1]+2)
                cond4 = (self.speed.km_h[i]<=self.speed.km_h[i-1] + 1) & \
                        (self.speed.km_h[i]<self.speed.km_h[i+1]-2)
                if (cond1 | cond2 | cond3 | cond4):
                    print i, cond1,cond2,cond3,cond4
                    break
                    self.speed.km_h[i] = 0.5*(self.speed.km_h[i-1]+self.speed.km_h[i+1])
                    
                    v = [self.coordinates.x[i+2] - self.coordinates.x[i],self.coordinates.y[i+2] - self.coordinates.y[i]]
                    norm = numpy.linalg.norm(v)
                    v[0]/=norm
                    v[1]/=norm
                    
                    self.coordinates.x[i+1] = (self.coordinates.x[i] + self.speed.km_h[i]/3.6*v[0])
                    self.coordinates.y[i+1] = (self.coordinates.y[i] + self.speed.km_h[i]/3.6*v[1])
                                      
                    errors+=1

            if ((errors==0) | (iteration>10)):
                break
                
    def RemoveOutliers22(self):
        n = self.acceleration.shape[0]
        errors = 0
        for i in range(1,n-1):
            if 0.5*(self.Distance2(self.acceleration.km_h[i],self.speed.km_h[i-1]) +\
                    self.Distance2(self.acceleration.km_h[i],self.speed.km_h[i+1]))<=\
                    self.Distance2(self.acceleration.km_h[i-1],self.speed.km_h[i+1]):
                errors+=1
                self.coordinates.x[i+1] = 0.5*(self.coordinates.x[i]+self.coordinates.x[i+2])
                self.coordinates.y[i+1] = 0.5*(self.coordinates.y[i]+self.coordinates.y[i+2])
                self.speed.km_h[i] = 3.6*self.Distance(self.coordinates.x[i+1],self.coordinates.y[i+1],self.coordinates.x[i],self.coordinates.y[i])
                self.speed.km_h[i+1] = 3.6*self.Distance(self.coordinates.x[i+1],self.coordinates.y[i+1],self.coordinates.x[i+2],self.coordinates.y[i+2])

# <codecell>

trips = []

driverID = 20
driverDir = '../Kaggle/drivers/'+str(driverID)
tripFiles = os.listdir(driverDir)

for index, tripFile in enumerate(tripFiles):
    df = pd.read_csv(driverDir+'/' + tripFile)
    trips.append(Trip(1,driverDir+'/' + tripFile,df))  

    if index>0:
        break

# <codecell>

trips[1].acceleration.head()

# <codecell>

trips[1].speed.head()

# <codecell>

trips[1].coordinates.ix[:6,:]

# <codecell>

tripInd = 1
start = 500
end = 600#trips[tripInd].n

# <codecell>

ggplot(trips[tripInd].coordinates.ix[start:end,:],aes(x='x',y='y')) + geom_point(size=5)

# <codecell>

ggplot(trips[tripInd].speed.ix[start:end,:],aes(x='ind',y='km_h')) + geom_point(size=10)#+\
#geom_point(trips[tripInd].centracceleration.ix[start:end,:],aes(x='ind',y='acc'),color='green',size=10)

# <codecell>

ggplot(trips[tripInd].centracceleration.ix[start:end,:],aes(x='ind',y='acc')) + geom_point(color='green',size=10)+\
geom_point(trips[tripInd].acceleration.ix[start:end,:],aes(x='ind',y='acc'),color='red',size=10)

# <codecell>

del df
df = trips[2].centracceleration.copy()
df.ind = trips[2].acceleration.acc
ggplot(df,aes(x='ind',y='acc')) + geom_point(size=10)

# <codecell>

del df
df = trips[0].centracceleration.copy()
df.ind = trips[0].acceleration.acc
ggplot(df,aes(x='ind',y='acc')) + geom_point(size=10)

# <codecell>


