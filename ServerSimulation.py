# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as  pd
from ggplot import *
df = pd.DataFrame(columns=['Day','Pnl','SharesTraded','OrdersTraded','ValueTraded','Pps'])

# <codecell>

d='/home/user1/Desktop/Share2Windows/'

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
df=df[df.SharesTraded>0]

# <codecell>

ggplot(df1,aes('Day','Pnl')) + geom_point(alpha=0.5) + \
stat_smooth(colour='green', span=0.5) + \
ggtitle('Stock position < 1000 shares: pnl = %s, mean = %s' % \
        (df1.Pnl.sum(),df1.Pnl.mean()))

# <codecell>

ggplot(df,aes('Day','Pnl')) + geom_point(alpha=0.5) + \
stat_smooth(colour='green', span=0.3) + \
ggtitle('Stock position < USD50K & smaller latency: pnl = %s, mean = %s' % \
        (df.Pnl.sum(),df.Pnl.mean()))

# <codecell>

df_merged = df1.merge(df, on='Day')
diff_df = pd.DataFrame()
diff_df['Day'] = df_merged.Day
diff_df['Pnl'] = df_merged.Pnl_y - df_merged.Pnl_x

# <codecell>

ggplot(diff_df,aes('Day','Pnl')) + geom_point(alpha=0.5) + stat_smooth(colour='green', span=0.5) + \
ggtitle('Difference of two simulations: pnl = %s, mean = %s' % \
        (df.Pnl.sum()-df1.Pnl.sum(),df.Pnl.mean()-df1.Pnl.mean()))

# <codecell>


