# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as  pd
from ggplot import *

# <codecell>

#Benchmark
df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/B1Lot/'
import os
for f_name in os.listdir(d):
    file = open(d+f_name, 'r')
    for line in file:
        words = line.split()
        if (len(words)>5) & (words[0]=='Test'):
            day = datetime.datetime.strptime(words[2].strip('><'),'%Y-%m-%d')
            pnl = float(words[4].replace(',',''))
            valueTraded = float(words[-2].replace(',',''))*0.5
            pps = float(words[-1].replace(',',''))
            shares = float(words[6].replace(',',''))*0.5
            orders = float(words[7].replace(',',''))*0.5
            df=df.append({'Day':day,'Pnl':pnl,'SharesTraded':shares,\
                          'OrdersTraded':orders,'ValueTraded':valueTraded,'Pps':pps},ignore_index=True)
            print '%s %s %s' % (day,pnl,valueTraded)
            break;
    file.close()
df=df[df.SharesTraded>0]
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + \
stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % \
        (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

#RandomForest original
df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/RF1Lot/'
import os
for f_name in os.listdir(d):
    file = open(d+f_name, 'r')
    for line in file:
        words = line.split()
        if (len(words)>5) & (words[0]=='Test'):
            day = datetime.datetime.strptime(words[2].strip('><'),'%Y-%m-%d')
            pnl = float(words[4].replace(',',''))
            valueTraded = float(words[-2].replace(',',''))*0.5
            pps = float(words[-1].replace(',',''))
            shares = float(words[6].replace(',',''))*0.5
            orders = float(words[7].replace(',',''))*0.5
            df=df.append({'Day':day,'Pnl':pnl,'SharesTraded':shares,\
                          'OrdersTraded':orders,'ValueTraded':valueTraded,'Pps':pps},ignore_index=True)
            print '%s %s %s' % (day,pnl,valueTraded)
            break;
    file.close()
df=df[df.SharesTraded>0]
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + \
stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % \
        (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

#RandomForest with expected target and rehedge before 9.30
df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/RF1LotRehedge/'
import os
for f_name in os.listdir(d):
    file = open(d+f_name, 'r')
    for line in file:
        words = line.split()
        if (len(words)>5) & (words[0]=='Test'):
            day = datetime.datetime.strptime(words[2].strip('><'),'%Y-%m-%d')
            pnl = float(words[4].replace(',',''))
            valueTraded = float(words[-2].replace(',',''))*0.5
            pps = float(words[-1].replace(',',''))
            shares = float(words[6].replace(',',''))*0.5
            orders = float(words[7].replace(',',''))*0.5
            df=df.append({'Day':day,'Pnl':pnl,'SharesTraded':shares,\
                          'OrdersTraded':orders,'ValueTraded':valueTraded,'Pps':pps},ignore_index=True)
            print '%s %s %s' % (day,pnl,valueTraded)
            break;
    file.close()
df=df[df.SharesTraded>0]
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + \
stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % \
        (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

#Added expected_pnl/variance threshold at 0.1
df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/Exp_Std_Simul/'
import os
for f_name in os.listdir(d):
    file = open(d+f_name, 'r')
    for line in file:
        words = line.split()
        if (len(words)>5) & (words[0]=='Test'):
            day = datetime.datetime.strptime(words[2].strip('><'),'%Y-%m-%d')
            pnl = float(words[4].replace(',',''))
            valueTraded = float(words[-2].replace(',',''))*0.5
            pps = float(words[-1].replace(',',''))
            shares = float(words[6].replace(',',''))*0.5
            orders = float(words[7].replace(',',''))*0.5
            df=df.append({'Day':day,'Pnl':pnl,'SharesTraded':shares,\
                          'OrdersTraded':orders,'ValueTraded':valueTraded,'Pps':pps},ignore_index=True)
            print '%s %s %s' % (day,pnl,valueTraded)
            break;
    file.close()
df=df[df.SharesTraded>0]
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + \
stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % \
        (df.Pnl.sum(),df.Pps.sum()))

# <codecell>


