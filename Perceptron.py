# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import random
import os, subprocess
 
class Perceptron:
    def __init__(self, N, G=None):
        # Random linearly separated data
        xA,yA,xB,yB = [random.uniform(-1, 1) for i in range(4)]
        self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        if G==None:
            G = lambda x: int(np.sign(self.V.T.dot(x)))            
        self.G = G
        self.X,self.onlyX, self.y = self.generate_points(N)
        self.N = N
        
 
    def generate_points(self, N, F=None):
        X = []
        onlyX = np.zeros((N,3))
        y = np.zeros(N)
        for i in range(N):
            x1,x2 = [random.uniform(-1, 1) for j in range(2)]
            x = np.array([1,x1,x2])
            
            if (F==None):
                F = self.G
                
            targetF=F(x)
              
            #Randomise    
            #if (rand(1)>0.9):
            #    targetF=-targetF
                
            onlyX[i,:]=x
            y[i]=targetF            
            X.append((x, targetF))
            
        return X, onlyX, y
    
    def transformX(self,newDim,F):
        transformedX = np.zeros((self.N,newDim))
        for i in range(self.N):
            transformedX[i,:]=F(self.onlyX[i,:])
        return transformedX
 
    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5,5))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        V = self.V
        a, b = -V[1]/V[2], -V[0]/V[2]
        l = np.linspace(-1,1)
        plt.plot(l, a*l+b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s]+'o')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'.')
        if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)),str(len(mispts))))
            plt.savefig('p_N%s' % (str(len(self.X))), \
                        dpi=200, bbox_inches='tight')
 
    def classification_error(self, F=None, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        if (F==None):
            F = self.G
            
        for x,s in pts:
            if F(x) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        return error
 
    def choose_miscl_point(self, vec):
        # Choose a random point among the misclassified
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]
 
    def pla(self, save=False, w = np.zeros(3)):
        # Initialize the weigths to zeros
        #w = np.zeros(3)
        X, N = self.X, len(self.X)
        it = 0
        # Iterate until all points are correctly classified
        F = lambda x: int(np.sign(w.T.dot(x)))
        while self.classification_error(F) != 0:
            it += 1
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s*x
            if save:
                self.plot(vec=w)
                plt.title('N = %s, Iteration %s\n' \
                          % (str(N),str(it)))
                plt.savefig('p_N%s_it%s' % (str(N),str(it)), \
                            dpi=200, bbox_inches='tight')
        self.it=it
        self.w = w
        
    def regress(self, onlyX=None):
        if onlyX==None:
            self.rw = (linalg.inv(matrix(self.onlyX.T)*matrix(self.onlyX))*matrix(self.onlyX.T))*\
            matrix(self.y).T
        else:
            return (linalg.inv(matrix(onlyX.T)*matrix(onlyX))*matrix(onlyX.T))*matrix(self.y).T
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)

# <codecell>

p = Perceptron(100)
p.plot()
p.pla()

# <codecell>

for k in range(10):
    i = randint(len(p.X))
    tmp = p.X[i]
    tmp = (tmp[0],-tmp[1])
    p.X[i] = tmp
    p.pla()
    print p.classification_error()
    tmp = p.X[i]
    tmp = (tmp[0],-tmp[1])
    p.X[i] = tmp

# <codecell>

i = randint(len(p.X))

# <codecell>

i

# <codecell>

tmp = p.X[i]
tmp = (tmp[0],-tmp[1])
p.X[i] = tmp

# <codecell>

p.pla()

# <codecell>

print p.classification_error()
tmp = p.X[i]
tmp = (tmp[0],-tmp[1])
p.X[i] = tmp

# <codecell>

print p.classification_error()

# <codecell>

type(newX[0])

# <codecell>

res = np.zeros(1000)
for i in range(1000):
    p = Perceptron(100)
    p.regress()
    G = lambda x: int(np.sign(p.rw.T.dot(x))) 
    res[i]=p.classification_error(G)
print res.mean()

# <codecell>

p.plot()

# <codecell>

rw = p.rw
G = lambda x: int(np.sign(rw.T.dot(x))) 
pG = p.G

# <codecell>

p.classification_error(G)

# <codecell>

p.classification_error(pG)

# <codecell>

p = Perceptron(1000,pG)
p.plot()

# <codecell>

p.classification_error(G)

# <codecell>

res = np.zeros(1000)
for i in range(1000):
    p = Perceptron(1000,pG)
    res[i]=p.classification_error(G)
print res.mean()

# <codecell>

a=np.array([0,0.001,0.01,0.1,0.5])

# <codecell>

argmin(np.abs(a-0.046884))

# <codecell>

res = np.zeros(1000)
for i in range(1000):
    p = Perceptron(10)
    p.regress()
    p.pla(w=np.array(p.RV.ravel())[0])
    res[i]=p.it
print res.mean()

# <codecell>

abs(a).argmin()

# <codecell>

res = np.zeros(1000)
for i in range(1000):
    p = Perceptron(1000)
    p.regress()
    res[i]=p.classification_error(p.rw)
print res.mean()

# <codecell>


  

# <codecell>

p.classification_error(RegressF)

# <codecell>


# <codecell>


# <codecell>


# <codecell>

G = lambda x: int(np.sign(x[1]*x[1]+x[2]*x[2]-0.6))
Tranformation = lambda x: [1,x[1],x[2],x[1]*x[2],x[1]*x[1],x[2]*x[2]]

ga = lambda x: int(np.sign(-1 - 0.05*x[1]+0.08*x[2]+0.13*x[1]*x[2]+1.5*(x[1]*x[1]+x[2]*x[2])))
gb = lambda x: int(np.sign(-1 - 0.05*x[1]+0.08*x[2]+0.13*x[1]*x[2]+1.5*(x[1]*x[1]+10*x[2]*x[2])))
gc = lambda x: int(np.sign(-1 - 0.05*x[1]+0.08*x[2]+0.13*x[1]*x[2]+1.5*(10*x[1]*x[1]+x[2]*x[2])))
gd = lambda x: int(np.sign(-1 - 1.5*x[1]+0.08*x[2]+0.13*x[1]*x[2]+0.05*(x[1]*x[1]+x[2]*x[2])))
ge = lambda x: int(np.sign(-1 - 0.05*x[1]+1.5*x[2]+0.13*x[1]*x[2]+0.15*(x[1]*x[1]+x[2]*x[2])))

# <codecell>

res = np.zeros((10,5))
for i in range(10):
    p = Perceptron(1000,G)
    RegressF = lambda x: int(np.sign(p.regress(p.transformX(6,Tranformation)).T.dot(Tranformation(x))))
    for j in range(p.N):
        p.X[j]=(p.X[j][0],RegressF(Tranformation(p.X[j][0])))
    res[i,:]=[p.classification_error(ga),p.classification_error(gb),p.classification_error(gc),\
              p.classification_error(gd),p.classification_error(ge)]

# <codecell>

res.mean(axis=0)

# <codecell>

res = np.zeros(100)
for i in range(100):
    p = Perceptron(1000,G)
    res[i]=p.classification_error(ga)
print res.mean()

# <codecell>


