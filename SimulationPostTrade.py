# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os

import numpy
import pandas as  pd
import math

from ggplot import *

import qimbs

# <codecell>

df = pd.DataFrame(columns=['Symbol','Shares','Pnl'])

# <codecell>

d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new_w15_longlatency_50kposition/'
#d='/home/user1/Desktop/Share2Windows/gab_20k_shares_0.1spread/'
reports = os.listdir(d)
#import stocks pnl

for i,f_name in enumerate(reports):
    day=f_name.split('_')[1]
    file = open(d+f_name, 'r')
    lineInd=0
    dropped = False
    activated = False
    for line in file:      
        if dropped:
            continue
        
        words = line.split()
        
        if len(words)==1:
            if words[0]=='!':
                break
                
        if len(words)==1:
            if words[0]=='Orders:':
                break    
                
        if activated:
            
            words = line.split(';')
            
            if len(words)==1:
                break
                
            if len(words)==10:
                df=df.append({'Symbol':words[0].strip(),'Shares':0.5*float(words[2]),'Pnl':float(words[5].replace(',',''))},ignore_index=True)
        
            continue
        
        if lineInd==4:
            if [w!='Qimb' for w in words].count(False)==0:
                dropped = True
                
        if len(words)==1:
            if words[0]=='Positions:':
                activated = True           
            
        lineInd+=1
    file.close()

# <codecell>

df = df.groupby(df.Symbol).sum()
df['Pps'] = df.Pnl/df.Shares
df=df.sort('Pps')

# <codecell>

ggplot(df,aes('Shares','Pps')) + geom_point(size = 5) + ggtitle("All data")

# <codecell>

ggplot(df[df.Shares<5000],aes('Shares','Pps')) + geom_point(size = 5) + ggtitle("Zoomed: Shares<5000")

# <codecell>

df

# <codecell>


