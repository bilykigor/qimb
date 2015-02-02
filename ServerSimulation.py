# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as  pd
from ggplot import *

# <codecell>

def importData(df,d):
    import os
    for f_name in os.listdir(d):
        file = open(d+f_name, 'r')
        for line in file:
            words = line.split()
            if (len(words)>5) & (words[0]=='Test'):
                day = datetime.datetime.strptime(words[2].strip('><'),'%Y-%m-%d')
                pnl = float(words[4].replace(',',''))
                valueTraded = float(words[-2].replace(',',''))#*0.5
                pps = float(words[-1].replace(',',''))
                shares = float(words[6].replace(',',''))*0.5
                orders = float(words[7].replace(',',''))*0.5
                df=df.append({'Day':day,'Pnl':pnl,'SharesTraded':shares,\
                              'OrdersTraded':orders,'ValueTraded':valueTraded,'Pps':pps},ignore_index=True)
                print '%s %s %s' % (day,pnl,valueTraded)
                break;
        file.close()
    df=df[df.SharesTraded>0]
    return df

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/B/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

#RandomForest original
df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/RF1Lot/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

#RandomForest with expected target and rehedge before 9.30
df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/RF1LotRehedge/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

#Added expected_pnl/variance threshold at 0.1
df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/Exp_Std_Simul/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/RF/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/B+RF/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/RF-Etf/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/RF-Etf+EVAR(1)/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/RF-Etf+EV(0.01)+EVAR(1)/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, pps = %s' % (df.Pnl.sum(),df.Pps.sum()))

# <codecell>

#
#
#
#
#
#

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new_w15/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

#
#This is Benchmark
#
df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new_w15_longlatency/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new_w15_longlatency_PCprobPosNeg/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/Qimb/Simulations/qimb_100shares_0.1spread_new_w15_longlatency_PCprobPosNegAllImb/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/Qimb/Simulations/qimb_100shares_0.1spread_new_w15_longlatency_B/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/Qimb/Simulations/qimb_100shares_0.1spread_new_w15_longlatency_B_AllImb/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new_w15_longlatency_PC/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new_w15_longlatency_ADV/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new_w15_longlatency_AVGP/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new_w15_longlatency_AVGP2/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/qimb_100shares_0.1spread_new_w15_longlatency_50kposition/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

#
#
#
#
#
#
#
#

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/gab_100shares_0.1spread/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/gab_100shares_0.1spread_new/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/gab_100shares_0.1spread_new_w15/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/gab_5k_shares_0.1spread/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/gab_5k_shares_0.1spread_w15/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/gab_20k_shares_0.1spread/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/gab_20k_shares_min1k_shares_0.1spread/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>

df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])
d='/home/user1/Desktop/Share2Windows/gab_20k_shares_min500_shares_0.1spread/'
df=importData(df,d);
ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.3) + \
ggtitle('pnl = %s, avg_pps = %s, avg_Exp = %s' % (df.Pnl.sum(),df.Pps.mean(),df.ValueTraded.mean()))

# <codecell>


