# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

rand(10)

# <codecell>


# <codecell>

for i in range(10):
    print randint(0,1)

# <codecell>

class Coin:
    from random import randint
    lastFlip = 0;
    last_nFlip = 0;
    def flip(self):
        self.lastFlip=randint(0,1)
        return self.lastFlip
    def flip_n_times(self,n):
        res = np.zeros(n)
        for i in range(n):
            res[i]=randint(0,1)
            
        self.last_nFlip = res 
        return res

# <codecell>

coin = Coin()
MinFreq= np.zeros(1000)

for j in range(1000):
    flipResults = np.zeros((1000,10))
    for i in range(1000):
        flipResults[i,:]=coin.flip_n_times(10)
    minFreq = min(sum(flipResults,axis=1))/10
    MinFreq[j] = minFreq

print MinFreq.mean()

# <codecell>


