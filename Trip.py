# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
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
    
    def __init__(self,dID,tID,df,window=1):
        self.window=window
        self.driverID = dID
        self.coordinates = df
        self.n = df.shape[0]
        self.ID = tID
        #self.DropPrecision()
        #self.RemoveDirectionOutliers()
        #self.getRadius()
        #self.RemoveRadiusOutliers()
        #self.getSpeed()
        #self.RemoveSpeedOutliers()
        #self.getAcc()
        #self.getFeatures()
        #self.getQuantiles()
        
    def getTripLen(self):
        self.n = self.coordinates.shape[0]
        self.tripDuration = (self.n-1)/60
        l = 0.0
        for i in range(1,self.n):
            l+=self.DistanceInd(i-1,1)
        self.tripLen = l
        
    def Quantiles(self,X):
        return X.quantile([x*0.02 for x in range(0,51)])
    
    def getQuantiles(self):
        self.speedQuantiles = self.speed.val.quantile([x*0.02 for x in range(0,51)])
        self.accQuantiles = self.acc.val.quantile([x*0.02 for x in range(0,51)])
        
    def getAcc(self):
        x=np.asarray(self.speed.val)
        self.acc = pd.DataFrame(x[self.window:]-x[:-self.window])
        self.acc.columns = ['val']
        
        x=np.asarray(self.acc.val)
        self.acc2 = pd.DataFrame(x[self.window:]-x[:-self.window])
        self.acc2.columns = ['val']
        
    def getCacc(self):
        self.cacc = \
        pd.DataFrame(np.asarray(pd.rolling_mean(self.speed.val,window=2).ix[1:])**2\
                     /np.asarray(self.radius.m),columns=['val'])
        
        
    def getSpeed(self):
        x=np.asarray(self.coordinates.x)
        y=np.asarray(self.coordinates.y)
        self.speed = pd.DataFrame(\
        np.sqrt((x[self.window:]-x[:-self.window])**2 + (y[self.window:]-y[:-self.window])**2)/self.window)
        self.speed.columns = ['val']
    
    def getFeatures(self): 
        #print 1
        self.features = pd.DataFrame(self.acc2.copy())
        self.features.columns = ['acc2']
        self.features['acc'] = np.asarray(pd.rolling_mean(self.acc.val,window=2).ix[1:])
        #self.features['cacc'] = np.asarray(pd.rolling_mean(self.cacc.val,window=2).ix[1:])
        self.features['v'] = np.asarray(pd.rolling_mean(self.speed.val,window=3).ix[2:])

    
    def Distance(self,x1,y1,x2,y2):
        return pow(pow(x1-x2,2)+pow(y1-y2,2),0.5)
    
    def DistanceInd(self,i,d):
        return pow(pow(self.coordinates.x[i+d]-self.coordinates.x[i],2)+\
                   pow(self.coordinates.y[i+d]-self.coordinates.y[i],2),0.5)
   
    def Radius(self,i):
        a = self.DistanceInd(i,1)
        b = self.DistanceInd(i+1,1)
        c = self.DistanceInd(i,2)
        
        p = 0.5*(a+b+c)
        denom = p*(p-a)*(p-b)*(p-c)
        if denom<=0:
            return np.inf
        else:
            return 0.25*a*b*c/pow(denom,0.5)
    
    def Turn(self,w,v):
        norm = np.linalg.norm(v)
        m= np.matrix(np.zeros((2,2)))
        m[0,0] = v.x/norm;m[0,1] = -v.y/norm
        m[1,0] = v.y/norm;m[1,1] = v.x/norm
        [xn,yn] = m.T*np.matrix(w).T
        return xn,yn
    
    def DropPrecision(self):
        self.n = self.coordinates.shape[0]
        for i in range(self.n-1):
            if self.DistanceInd(i,1)<1.5:
                #print i
                self.coordinates.x[i+1] = self.coordinates.x[i]
                self.coordinates.y[i+1] = self.coordinates.y[i]
        
    
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
                    #print i
                    self.coordinates.x[i+2] = 0.5*(self.coordinates.x[i+3]+self.coordinates.x[i+1])
                    self.coordinates.y[i+2] = 0.5*(self.coordinates.y[i+3]+self.coordinates.y[i+1])
                    errors+=1

            if ((errors==0) | (iteration>10)):
                break
                
    def getRadius(self):
        n = self.n-2
        self.radius = pd.DataFrame(np.zeros((n,1)))
        self.radius.columns = ['m']
        
        for i in range(n):
            self.radius.m[i] = self.Radius(i)
                
    def RemoveRadiusOutliers(self):
        n = self.n-2
        iteration = 0;
        
        while True:
            iteration+=1
            errors = 0
            for i in range(n):                
                if self.radius.m[i]==np.inf:
                    continue
                    
                if (self.radius.m[i]<2):# | (self.radius.m[i]>500)):
                    #print i
                    self.coordinates.x[i+1] = 0.5*(self.coordinates.x[i]+self.coordinates.x[i+2])
                    self.coordinates.y[i+1] = 0.5*(self.coordinates.y[i]+self.coordinates.y[i+2])
                    self.radius.m[i] = inf
                    if i>0:
                        self.radius.m[i-1] = self.Radius(i-1)
                    if i<n-1:
                        self.radius.m[i+1] = self.Radius(i+1)

                    errors+=1
  
            if ((errors==0) | (iteration>100)):
                break
    
    def RemoveSpeedOutliers(self):
        n = self.n-1
        iteration = 0;
        move = 0.1
        while True:
            iteration+=1
            errors = 0
            for i in range(1,n-1):
                cond1 = (self.speed.val[i]>self.speed.val[i-1] + 2.0/3.6) & \
                        (self.speed.val[i]>=self.speed.val[i+1] - 1.0/3.6)
                cond2 = (self.speed.val[i]<self.speed.val[i-1] - 2.0/3.6) & \
                        (self.speed.val[i]<=self.speed.val[i+1] + 1.0/3.6)
                cond3 = (self.speed.val[i]>=self.speed.val[i-1] - 1.0/3.6) & \
                        (self.speed.val[i]>self.speed.val[i+1] + 2.0/3.6)
                cond4 = (self.speed.val[i]<=self.speed.val[i-1] + 1.0/3.6) & \
                        (self.speed.val[i]<self.speed.val[i+1] - 2.0/3.6)
                    
                if (cond1 | cond3):
                    #print i
                    v = [self.coordinates.x[i] - self.coordinates.x[i+1],\
                         self.coordinates.y[i] - self.coordinates.y[i+1]]
                    norm = np.linalg.norm(v)
                    if norm==0:
                        continue
                    v[0]/=norm
                    v[1]/=norm
                    
                    self.coordinates.x[i+1] += move*v[0]
                    self.coordinates.y[i+1] += move*v[1]
                    
                    self.speed.val[i] = self.DistanceInd(i,1)
                    self.speed.val[i+1] = self.DistanceInd(i+1,1)
                    
                    errors+=1
                    
                if (cond2 | cond4):
                    #print i
                    v = [self.coordinates.x[i-1] - self.coordinates.x[i],\
                         self.coordinates.y[i-1] - self.coordinates.y[i]]
                    norm = np.linalg.norm(v)
                    if norm==0:
                        continue
                    v[0]/=norm
                    v[1]/=norm
                    
                    self.coordinates.x[i] += move*v[0]
                    self.coordinates.y[i] += move*v[1]
                                   
                    self.speed.val[i-1] = self.DistanceInd(i-1,1)
                    self.speed.val[i] = self.DistanceInd(i,1)
                    
                    errors+=1
                    
                if (cond2 | cond4 | cond1 | cond3):     
                    for j in range(max(1,i-3),min(i+3,n-1)):
                        #print j
                        r = self.Radius(j-1)
                        self.radius.m[j-1] = r

            if ((errors==0) | (iteration>10)):
                break
    

# <codecell>


