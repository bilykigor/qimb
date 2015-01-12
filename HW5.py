# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

e = lambda n, sigma, d:  sigma*sigma*(1-(d+1)/n)

# <codecell>

x=[10,25,100,500,1000]
sigma = .1
d=8
for v in x:
    print v
    print sigma*sigma*(1-(d+1)/v)>0.008

# <codecell>

e(0,0,0)

# <codecell>


sigma*sigma*(1-(d+1)/n)

