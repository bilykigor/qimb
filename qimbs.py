# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from ggplot import *

# <codecell>

def import_month(month):
    import os.path
    df=pd.DataFrame()
    
    if month<10:
        m = '0%s' % month
    else:
        m = '%s' % month
    
    #Read data for the month
    for i in range (1,32):  
        if i<10:
            d = '0%s' % i
        else:
            d = '%s' % i
        
        f = '/home/user1/PyProjects/data_old/2014-%s-%s/CommonAggr_2014-%s-%s.csv' % (m,d,m,d)
        
        if (not os.path.isfile(f)): continue
            
        if (df.shape[0]==0):
            df = pd.read_csv(f, low_memory=False)
        else:
            df=df.append(pd.read_csv(f, low_memory=False))
    
    #Set target columns
    target_cols = ['Date','Time','Fracs','Symbol','Reason','tSide','tPrice','tShares',
                    'Bid_P', 'Bid_S', 'Ask_P', 'Ask_S', 
                    'ImbRef','ImbCBC', 'ImbFar', 'ImbShares', 'ImbPaired']

    #Get target columts from the data
    df = df[target_cols]
    df.index = range(df.shape[0])
    
    return df

def import_month2(month):
    import os.path
    df=pd.DataFrame()
    
    if month<10:
        m = '0%s' % month
    else:
        m = '%s' % month
    
    #Read data for the month
    for i in range (1,32):  
        if i<10:
            d = '0%s' % i
        else:
            d = '%s' % i
        
        f = '/home/user1/PyProjects/data/AggrCommon_2014-%s-%s.csv' % (m,d)
        
        if (not os.path.isfile(f)): continue
            
        if (df.shape[0]==0):
            df = pd.read_csv(f, low_memory=False)
        else:
            df=df.append(pd.read_csv(f, low_memory=False))
    
    #Set target columns
    target_cols = ['Date','Time','Fracs','Symbol','Reason','tType','tVenue','tSide','tPrice','tShares',
                    'Bid_P', 'Bid_S', 'Ask_P', 'Ask_S', 'nsdq_BP', 'nsdq_BS', 'nsdq_AP', 'nsdq_AS',
                    'ImbRef','ImbCBC', 'ImbFar', 'ImbShares', 'ImbPaired', 'Bid2', 'Ask2']

    #Get target columts from the data
    df = df[target_cols]
    df.index = range(df.shape[0])
    
    return df

def import_file(f):
    df = pd.read_csv('/home/user1/PyProjects/data/' + f, low_memory=False)
    target_cols = ['Date','Time','Fracs','Symbol','Reason','tSide','tPrice','tShares',
                    'Bid_P', 'Bid_S', 'Ask_P', 'Ask_S', 
                    'ImbRef','ImbCBC', 'ImbFar', 'ImbShares', 'ImbPaired']

    df = df[target_cols]
    df.index = range(df.shape[0])
    
    return df
  
def create_timestamp(df):
    Timestamp = []
    for i in range(df.shape[0]):
        Timestamp.append(datetime.datetime.strptime(df.Date[i] +' '+df.Time[i]+' '+df.Fracs[i][0:3]+df.Fracs[i][4:7],'%Y-%m-%d %H:%M:%S %f'))

    df['Timestamp'] = Timestamp
    df = df.set_index(['Timestamp'])
    df = df.drop(['Date','Time','Fracs'],1)
    
    return df

def sigmoid(x):
    import numpy
    return 1/(1+numpy.exp(-x))

# <codecell>

#Visualization of features
from sklearn import preprocessing
def visualize(fdf,Features,f,binwidth,scaled):
    fdf_new = fdf.copy()
    if scaled:
        fdf_new[f] = preprocessing.scale(fdf_new[f])
    g=ggplot(fdf_new[fdf_new.Move<0],aes(x=f)) + \
    geom_histogram(color='red',binwidth = binwidth,alpha=0.25,\
                   fill = 'red') + \
    geom_histogram(fdf_new[fdf_new.Move>0],aes(x=f), \
                   color='green',fill = 'green',\
                   binwidth = binwidth,alpha=0.25) + \
    ggtitle(Features[f]) 
    #xlim(-1,1)+ ylim(-20,20)
    return g

# <codecell>

def get_imbelanceMSG(df,nImb):
    (startTime, endTime) = getImbTime(nImb)
    
    imbalanceMsg = df[df.Reason == 'Imbalance'].between_time(startTime,endTime)
    #.between_time('9:29:52','9:29:57')
    imbalanceMsg = imbalanceMsg[
    (imbalanceMsg.Bid_P>0.01) & 
    (imbalanceMsg.Ask_P<199999.99) & 
    (imbalanceMsg.ImbRef>0) & 
    (imbalanceMsg.ImbCBC>0) &
    (imbalanceMsg.ImbFar>0) &
    (imbalanceMsg.ImbShares!=0)
    ]

    imbalanceMsg = imbalanceMsg[['Symbol','Bid_P','Bid_S','Ask_P','Ask_S','ImbRef','ImbCBC','ImbFar','ImbShares','ImbPaired']]
    imbalanceMsg['Date'] = imbalanceMsg.index.date
    imbalanceMsg['Timestamp'] = imbalanceMsg.index
        
    #Getting additional info about previous day
    OPC = df[df.Reason == 'OPG']
    OPC = OPC[['Symbol','tPrice']]
    OPC.columns = ['Symbol','OPC_P']
    OPC['Date'] = OPC.index.date

    prev_OPC = df[df.Reason == 'OPG']
    prev_OPC = prev_OPC[['Symbol','tPrice','tShares']]
    prev_OPC.columns = ['Symbol','PrevOPC_P','PrevOPC_S']
    prev_OPC['Date'] = prev_OPC.index.date
    for i in range(prev_OPC.shape[0]):
        if prev_OPC.Date[i].weekday()==4:
            prev_OPC.Date[i]+=datetime.timedelta(days=3)
        else:
            prev_OPC.Date[i]+=datetime.timedelta(days=1)

    prev_CLC = df[df.tSide == 'YDAY']
    prev_CLC = prev_CLC[['Symbol','tPrice','tShares']]
    prev_CLC.columns = ['Symbol','PrevCLC_P','PrevCLC_S']
    prev_CLC['Date'] = prev_CLC.index.date

    #Adding prev day info to imbalance information
    imbalanceMsg = pd.merge(imbalanceMsg, OPC, on=['Symbol','Date'])
    imbalanceMsg = pd.merge(imbalanceMsg, prev_OPC, on=['Symbol','Date'])
    imbalanceMsg = pd.merge(imbalanceMsg, prev_CLC, on=['Symbol','Date'])

    #Filtering data with no prev OPC or prev CLC
    imbalanceMsg = imbalanceMsg[(imbalanceMsg.OPC_P>0) & (imbalanceMsg.PrevOPC_P>0)]
    imbalanceMsg.index = range(imbalanceMsg.shape[0])
    
    #Adding new feature which reflects price move direction
    imbalanceMsg['Move'] = imbalanceMsg.Bid_P
    imbalanceMsg.Move = 0
    imbalanceMsg.Move[imbalanceMsg.OPC_P>imbalanceMsg.Ask_P] = 1
    imbalanceMsg.Move[imbalanceMsg.OPC_P<imbalanceMsg.Bid_P] = -1
    
    return imbalanceMsg   
    
def create_features(imbalanceMsg):  
    #Creating features for algorithm
    import numpy
    fdf = pd.DataFrame()
    Features = dict()

    fdf['Symbol'] = imbalanceMsg.Symbol
    fdf['Date'] = imbalanceMsg.Date

    fdf['Move'] = imbalanceMsg.Move
    Features['Move'] = '1:OpenCross>Ask(9.28); -1:OpenCross<Bid(9.28); 0:otherwise'
    
    fdf['Pnl'] = (imbalanceMsg.Move==-1)*(imbalanceMsg.OPC_P-imbalanceMsg.Ask_P)+\
                 (imbalanceMsg.Move==1)*(imbalanceMsg.Bid_P-imbalanceMsg.OPC_P)

    fdf['Spread'] = (imbalanceMsg.Ask_P - imbalanceMsg.Bid_P)/imbalanceMsg.PrevCLC_P
    Features['Spread'] = '(Ask-Bid) at 9.28'

    fdf['D1'] = 100*(imbalanceMsg.PrevCLC_P/imbalanceMsg.PrevOPC_P-1)
    Features['D1'] = 'Asset growth a day before'

    fdf['D2'] = 100*(0.5*(imbalanceMsg.Ask_P + imbalanceMsg.Bid_P)/imbalanceMsg.PrevOPC_P-1)
    Features['D2'] = 'Mid(9.28)/OPC(day before)-1'

    fdf['D3'] = 100*(0.5*(imbalanceMsg.Ask_P + imbalanceMsg.Bid_P)/imbalanceMsg.PrevCLC_P-1)
    Features['D3'] = 'Mid(9.28)/CloseCross(day before)-1'

    
    
    fdf['D4'] = 100*(imbalanceMsg.Bid_P-imbalanceMsg.ImbRef)/imbalanceMsg.PrevCLC_P
    Features['D4'] = '(Bid(9.28)-ImbRef(9.28))/CloseCross(day before)'

    fdf['D5'] = 100*(imbalanceMsg.ImbRef-imbalanceMsg.Ask_P)/imbalanceMsg.PrevCLC_P
    Features['D5'] = '(ImbRef(9.28)-Ask(9.28))/CloseCross(day before)'
    
    
    fdf['D44'] = 100*(imbalanceMsg.Bid_P-imbalanceMsg.ImbRef)/imbalanceMsg.PrevOPC_P
    Features['D44'] = '(Bid(9.28)-ImbRef(9.28))/OpenCross(day before)'

    fdf['D55'] = 100*(imbalanceMsg.ImbRef-imbalanceMsg.Ask_P)/imbalanceMsg.PrevOPC_P
    Features['D55'] = '(ImbRef(9.28)-Ask(9.28))/OpenCross(day before)'
    
    
    fdf['D444'] = (imbalanceMsg.Bid_P-imbalanceMsg.ImbRef)/(1+imbalanceMsg.Ask_P - imbalanceMsg.Bid_P)
    Features['D444'] = '(Bid(9.28)-ImbRef(9.28))/1+Spread'

    fdf['D555'] = (imbalanceMsg.ImbRef-imbalanceMsg.Ask_P)/(1+imbalanceMsg.Ask_P - imbalanceMsg.Bid_P)
    Features['D555'] = '(ImbRef(9.28)-Ask(9.28))/1+Spread'
    

    fdf['D6'] = 100*(imbalanceMsg.ImbRef/imbalanceMsg.PrevOPC_P-1)
    Features['D6'] = 'ImbRef(9.28)/OpenCross(day before)-1'

    fdf['D7'] = 100*(imbalanceMsg.ImbRef/imbalanceMsg.PrevCLC_P-1)
    Features['D7'] = 'ImbRef(9.28)/CloseCross(day before)-1'
    
    fdf['D66'] = 100*(2*imbalanceMsg.ImbRef/(imbalanceMsg.Ask_P + imbalanceMsg.Bid_P)-1)
    Features['D66'] = 'ImbRef(9.28)/Mid-1'
    
    

    fdf['V1'] = (imbalanceMsg.Ask_S - imbalanceMsg.Bid_S)/(100*numpy.sign(imbalanceMsg.ImbShares)+imbalanceMsg.ImbShares)
    Features['V1'] = '(Ask_S-Bid_S) at 9.28/Imbalance(9.28)'
    
    fdf['V11'] = (imbalanceMsg.Ask_S - imbalanceMsg.Bid_S)/(100+imbalanceMsg.ImbPaired)
    Features['V11'] = '(Ask_S-Bid_S) at 9.28/PairedS(9.28)'

    fdf['V2'] = (imbalanceMsg.Ask_S - imbalanceMsg.Bid_S)/imbalanceMsg.PrevOPC_S
    Features['V2'] = '(Ask_S-Bid_S) at 9.28/OpenCross(day before)'

    fdf['V3'] = (imbalanceMsg.Ask_S - imbalanceMsg.Bid_S)/imbalanceMsg.PrevCLC_S
    Features['V3'] = '(Ask_S-Bid_S) at 9.28/CloseCross(day before)'

    fdf['V4'] = imbalanceMsg.ImbShares/imbalanceMsg.PrevOPC_S
    Features['V4'] = 'ImbalanceS(9.28)/OpenCrossS(day before)'

    fdf['V5'] = imbalanceMsg.ImbShares/imbalanceMsg.PrevCLC_S
    Features['V5'] = 'ImbalanceS(9.28)/CloseCrossS(day before)'

    fdf['V6'] = imbalanceMsg.ImbPaired/imbalanceMsg.PrevOPC_S
    Features['V6'] = 'PairedS(9.28)/OpenCrossS(day before)'

    fdf['V7'] = imbalanceMsg.ImbPaired/imbalanceMsg.PrevCLC_S
    Features['V7'] = 'PairedS(9.28)/CloseCrossS(day before)'
    
    fdf['V8'] = imbalanceMsg.ImbShares/(100+imbalanceMsg.ImbPaired)
    Features['V8'] = 'ImbalanceS(9.28)/PairedS(9.28)'
    
    fdf['V9'] = imbalanceMsg.PrevOPC_S/(100+imbalanceMsg.PrevCLC_S)
    Features['V9'] = 'OpenCrossS(day before)/CloseCrossS(day before)'


    fdf['a1'] = fdf['D1']*fdf['D2']
    Features['a1'] = Features['D1'] + ' Multiply ' + Features['D2']
    
    fdf['a2'] = fdf['D2']*fdf['D3']
    Features['a2'] = Features['D3'] + ' Multiply ' + Features['D2']
    
    fdf['a3'] = fdf['D3']*fdf['D4']
    fdf['a4'] = fdf['D5']*fdf['D4']
    Features['a4'] = Features['D5'] + ' Multiply ' + Features['D4']
    
    fdf['a5'] = fdf['D5']*fdf['D6']
    fdf['a6'] = fdf['D1']*fdf['D6']
    Features['a6'] = Features['D1'] + ' Multiply ' + Features['D6']
    
    fdf['a7'] = fdf['V1']*fdf['V2']
    Features['a7'] = Features['V1'] + ' Multiply ' + Features['V2']
    
    fdf['a8'] = fdf['V2']*fdf['V3']
    fdf['a9'] = fdf['V3']*fdf['V4']
    Features['a9'] = Features['V3'] + ' Multiply ' + Features['V4']
    
    fdf['a10'] = fdf['V5']*fdf['V4']
    fdf['a11'] = fdf['V5']*fdf['V6']
    Features['a11'] = Features['V5'] + ' Multiply ' + Features['V6']
    
    fdf['a12'] = fdf['V7']*fdf['V6']
    fdf['a13'] = fdf['V7']*fdf['V1']
    Features['a13'] = Features['V1'] + ' Multiply ' + Features['V7']
    
    fdf['a14'] = np.sign(imbalanceMsg.ImbShares)
    Features['a14'] = 'Sign of Imbalance'
    
    fdf.index = range(fdf.shape[0])
    
    return fdf, Features

def create_features2(imbalanceMsg):  
    #Creating features for algorithm
    import numpy
    fdf = pd.DataFrame()
    Features = dict()

    fdf['Symbol'] = imbalanceMsg.Symbol
    fdf['Date'] = imbalanceMsg.Date

    fdf['Move'] = imbalanceMsg.OPC_P/imbalanceMsg.ImbRef-1   
    Features['Move'] = 'OpenCross/RefPrice(9.28)-1'
    
    fdf['Pnl'] = imbalanceMsg.OPC_P-imbalanceMsg.ImbRef
    Features['Pnl'] = 'OpenCross/RefPrice(9.28)'
    
    fdf['Bid'] = imbalanceMsg.Bid_P/imbalanceMsg.ImbRef-1
    Features['Bid'] = 'Bid(9.28)'
    
    fdf['Ask'] = imbalanceMsg.Ask_P/imbalanceMsg.ImbRef-1
    Features['Ask'] = 'Ask(9.28)'
    
    fdf['Ref'] = imbalanceMsg.ImbRef
    Features['Ref'] = 'Ref(9.28)'
    
    fdf['Near'] = imbalanceMsg.ImbCBC/imbalanceMsg.ImbRef-1
    Features['Near'] = 'Near(9.28)'
    
    fdf['Far'] = imbalanceMsg.ImbFar/imbalanceMsg.ImbRef-1
    Features['Far'] = 'Far(9.28)'
    
    fdf['PrevOPC'] = imbalanceMsg.PrevOPC_P/imbalanceMsg.ImbRef-1
    Features['PrevOPC'] = 'PrevOPC'
    
    fdf['PrevCLC'] = imbalanceMsg.PrevCLC_P/imbalanceMsg.ImbRef-1
    Features['PrevCLC'] = 'PrevCLC'

    
    fdf.index = range(fdf.shape[0])
    
    return fdf, Features

def create_features3(imbalanceMsg):  
    #Creating features for algorithm
    import numpy
    fdf = pd.DataFrame()
    Features = dict()

    midP = 0.5*(imbalanceMsg.Ask_P + imbalanceMsg.Bid_P)
    bid = imbalanceMsg.Bid_P#nsdq_BP #
    bidS = imbalanceMsg.Bid_S#nsdq_BS#
    ref = imbalanceMsg.ImbRef
    ask = imbalanceMsg.Ask_P#nsdq_AP#
    askS = imbalanceMsg.Ask_S#nsdq_AS#
    near = imbalanceMsg.ImbCBC
    far = imbalanceMsg.ImbFar
    closeP = imbalanceMsg.PrevCLC_P
    
    fdf['Symbol'] = imbalanceMsg.Symbol
    fdf['Date'] = imbalanceMsg.Date

    fdf['Move'] = imbalanceMsg.Move
    Features['Move'] = 'Move'
       
    fdf['Bid'] = bid/ref-1
    Features['Bid'] = 'Bid(9.28)'
    
    fdf['Ask'] = ask/ref-1
    Features['Ask'] = 'Ask(9.28)'
       
    fdf['Near'] = near/ref-1
    Features['Near'] = 'Near(9.28)'
    
    fdf['Far'] = far/ref-1
    Features['Far'] = 'Far(9.28)'
        
    fdf['PrevCLC'] = closeP/ref-1
    Features['PrevCLC'] = 'PrevCLC'
    
    fdf['Spread'] = (ask - bid)/ref
    Features['Spread'] = '(Ask-Bid) at 9.28'

    fdf['D3'] = 100*(midP/closeP-1)
    Features['D3'] = 'Mid(9.28)/CloseCross(day before)-1'
    
    fdf['D4'] = 100*(bid-ref)/closeP
    Features['D4'] = '(Bid(9.28)-ImbRef(9.28))/CloseCross(day before)'

    fdf['D5'] = 100*(ref-ask)/closeP
    Features['D5'] = '(ImbRef(9.28)-Ask(9.28))/CloseCross(day before)'       
    
    fdf['D444'] = (bid-ref)/(1+ask - bid)
    Features['D444'] = '(Bid(9.28)-ImbRef(9.28))/1+Spread'

    fdf['D555'] = (ref-ask)/(1+ask - bid)
    Features['D555'] = '(ImbRef(9.28)-Ask(9.28))/1+Spread'
    
    fdf['D7'] = 100*(ref/closeP-1)
    Features['D7'] = 'ImbRef(9.28)/CloseCross(day before)-1'
    
    fdf['D66'] = 100*(ref/midP-1)
    Features['D66'] = 'ImbRef(9.28)/Mid-1'

    fdf['V1'] = (askS - bidS)/(100*numpy.sign(imbalanceMsg.ImbShares)+imbalanceMsg.ImbShares)
    Features['V1'] = '(Ask_S-Bid_S) at 9.28/Imbalance(9.28)'
    
    fdf['V11'] = (askS - bidS)/(100+imbalanceMsg.ImbPaired)
    Features['V11'] = '(Ask_S-Bid_S) at 9.28/PairedS(9.28)'
    
    fdf['V8'] = imbalanceMsg.ImbShares/(100+imbalanceMsg.ImbPaired)
    Features['V8'] = 'ImbalanceS(9.28)/PairedS(9.28)'
         
    fdf['a3'] = fdf['D3']*fdf['D4']
    
    fdf['a4'] = fdf['D5']*fdf['D4']
    Features['a4'] = Features['D5'] + ' Multiply ' + Features['D4']
            
    fdf['a14'] = np.sign(imbalanceMsg.ImbShares)
    Features['a14'] = 'Sign of Imbalance'
    
    fdf.index = range(fdf.shape[0])
    
    return fdf, Features

def create_features33(imbalanceMsg):  
    #Creating features for algorithm
    import numpy
    fdf = pd.DataFrame()

    midP = 0.5*(imbalanceMsg.Ask_P + imbalanceMsg.Bid_P)
    bid = imbalanceMsg.Bid_P
    bidS = imbalanceMsg.Bid_S    
    min3 = imbalanceMsg[['ImbRef','ImbCBC','ImbFar']].apply(min,axis=1)
    max3 = imbalanceMsg[['ImbRef','ImbCBC','ImbFar']].apply(max,axis=1)
    
    f = lambda x: int(x[3]>0)*max(x[0],x[1])+\
                  int(x[3]<0)*min(x[0],x[1])+\
                  int(x[3]==0)*(x[0]+x[1])*0.5
            
    f = lambda x:2.0*x[0]*x[1]/(x[0]+x[1])
    
    f = lambda x: int(x[3]>0)*(0.85*min(x[0],x[1])+0.15*max(x[0],x[1]))+\
                  int(x[3]<0)*(0.15*min(x[0],x[1])+0.85*max(x[0],x[1]))+\
                  int(x[3]==0)*(x[0]+x[1])*0.5
            
    f = lambda x:  (x[0] + x[1] +  0.5*(x[4] + x[5]))/3.0
    
    f = lambda x:  int(x[3]>0)*\
                   (
                   (min(x[0],x[1])> x[4])*(x[0]+x[1])*0.5+\
                   (min(x[0],x[1])<=x[4])*(0.85*min(x[0],x[1])+0.15*max(x[0],x[1])))+\
                   int(x[3]<0)*\
                   ((max(x[0],x[1])<x[4])*(x[0]+x[1])*0.5+\
                   (max(x[0],x[1])>=x[4])*(0.15*min(x[0],x[1])+0.85*max(x[0],x[1])))+\
                   int(x[3]==0)*(x[0]+x[1])*0.5
    
    imbalanceMsg['Mid_P'] = 0.5*( imbalanceMsg.Ask_P +  imbalanceMsg.Bid_P)
    ref = imbalanceMsg[['ImbRef','ImbCBC','ImbFar','ImbShares','Mid_P']].apply(f,axis=1)
    #ref = imbalanceMsg.ImbRef
    ask = imbalanceMsg.Ask_P
    askS = imbalanceMsg.Ask_S
    near = imbalanceMsg.ImbCBC
    far = imbalanceMsg.ImbFar
    closeP = imbalanceMsg.PrevCLC_P
    spread = ask - bid
    
    fdf['Symbol'] = imbalanceMsg.Symbol
    
    fdf['Date'] = imbalanceMsg.Date
        
    fdf['OPC_P'] = imbalanceMsg.OPC_P
    
    fdf['Ask_P'] = ask
    
    fdf['Bid_P'] = bid
    
    fdf['Mid_P'] = midP
    #---------------------------------
        
    fdf['Bid'] = ref/bid-1
    
    fdf['Ask'] = ask/ref-1   
        
    fdf['BidD'] = bid - ref
    
    fdf['AskD'] = ref - ask
    #---------------------------------
    
    fdf['Near'] = near/ref-1
    
    fdf['Far'] = far/ref-1
    
    fdf['Spread'] = spread/ref
    #---------------------------------
    
    fdf['D4'] = 100*(bid-ref)/closeP

    fdf['D5'] = 100*(ref-ask)/closeP
    #---------------------------------
        
    fdf['D444'] = (bid-ref)/(1+spread)

    fdf['D555'] = (ref-ask)/(1+spread)
    #---------------------------------
    
    fdf['D66'] = 100*(ref/midP-1)
    #---------------------------------
    
    fdf['V1'] = numpy.sign(imbalanceMsg.ImbShares)*(askS - bidS)/(100+np.abs(imbalanceMsg.ImbShares))
    
    fdf['V1n'] = numpy.sign(imbalanceMsg.ImbShares)*(askS - bidS)/((askS + bidS)/2+np.abs(imbalanceMsg.ImbShares))
    
    fdf['V11'] = numpy.sign(imbalanceMsg.ImbShares)*(askS - bidS)/(100+imbalanceMsg.ImbPaired)
    
    fdf['V11n'] =numpy.sign(imbalanceMsg.ImbShares)*(askS - bidS)/((askS + bidS)/2+imbalanceMsg.ImbPaired)
    #---------------------------------
    
    fdf['V8'] = imbalanceMsg.ImbShares/(100+imbalanceMsg.ImbPaired)
    
    fdf['V8n'] = imbalanceMsg.ImbShares/((askS + bidS)/2+imbalanceMsg.ImbPaired)
    
    fdf['V8nn'] = (imbalanceMsg.ImbShares-(askS - bidS))/((askS + bidS)/2+imbalanceMsg.ImbPaired)
    #---------------------------------
        
    fdf['a1'] = fdf['Bid']*fdf['Ask']

    fdf['a4'] = fdf['D5']*fdf['D4']
    
    fdf['a5'] = fdf['D444']*fdf['D555']
    #---------------------------------
       
    fdf['a14'] = np.sign(imbalanceMsg.ImbShares)
    
    fdf['y'] = 1*(imbalanceMsg.OPC_P>ask) -  1*(imbalanceMsg.OPC_P<bid)
        
    fdf['PrevCLC'] = closeP/ref-1   

    fdf['D3'] = 100*(midP/closeP-1)
        
    fdf['D7'] = 100*(ref/closeP-1)
    
    fdf['a3'] = fdf['D3']*fdf['D4']
       
    fdf['a6'] = fdf['D444']*fdf['D444']
    
    fdf['a7'] = fdf['D555']*fdf['D555']
    
    fdf['p'] = numpy.floor(fdf['Mid_P']/50)
    
    #fdf = fdf[max3-min3<0.5*ref]
        
    fdf.index = range(fdf.shape[0])
    
    return fdf

# <codecell>

def create_features4(imbalanceMsg):  
    #Creating features for algorithm
    import numpy
    fdf = pd.DataFrame()
    Features = dict()

    fdf['Symbol'] = imbalanceMsg.Symbol
    fdf['Date'] = imbalanceMsg.Date

    fdf['Move'] = imbalanceMsg.Move
    Features['Move'] = 'Move'
    
    fdf['CMove'] = imbalanceMsg.Move
    fdf.CMove = 0
    fdf.CMove[imbalanceMsg.OPC_P-imbalanceMsg.Ask_P>0.1] = 3
    fdf.CMove[(imbalanceMsg.OPC_P-imbalanceMsg.Ask_P>0.02) & 
              (imbalanceMsg.OPC_P-imbalanceMsg.Ask_P<=0.1)] = 2
    fdf.CMove[(imbalanceMsg.OPC_P-imbalanceMsg.Ask_P>0)
              & (imbalanceMsg.OPC_P-imbalanceMsg.Ask_P<=0.02)] = 1
    fdf.CMove[(imbalanceMsg.Bid_P-imbalanceMsg.OPC_P>0.1)] = -3
    fdf.CMove[(imbalanceMsg.Bid_P-imbalanceMsg.OPC_P>0.02)
              & (imbalanceMsg.Bid_P-imbalanceMsg.OPC_P<=0.1)] = -2
    fdf.CMove[(imbalanceMsg.Bid_P-imbalanceMsg.OPC_P>0)
              & (imbalanceMsg.Bid_P-imbalanceMsg.OPC_P<=0.02)] = -1
    Features['CMove'] = 'CMove'
    
    
    
    fdf['0'] = np.sign(imbalanceMsg.ImbShares)
    Features['0'] = 'Side'
       
    fdf['1'] = imbalanceMsg.Bid_P-(imbalanceMsg.Ask_P - imbalanceMsg.Bid_P)
    Features['1'] = 'Bid-Spread'
    
    fdf['2'] = imbalanceMsg.Bid_P
    Features['2'] = 'Bid'
       
    fdf['3'] = imbalanceMsg.Ask_P
    Features['3'] = 'Ask'
    
    fdf['4'] = imbalanceMsg.Ask_P + (imbalanceMsg.Ask_P - imbalanceMsg.Bid_P)
    Features['4'] = 'Ask+Spread'
    
    fdf['5'] = imbalanceMsg.ImbRef
    Features['5'] = 'Ref'
    
    fdf['6'] = imbalanceMsg.ImbCBC
    Features['6'] = 'Near'
    
    fdf['7'] = imbalanceMsg.ImbFar
    Features['7'] = 'Far'
    
    fdf['8'] = imbalanceMsg.PrevCLC_P
    Features['8'] = 'Close'
    
    fdf['n2'] = imbalanceMsg.nsdq_BP
    Features['n2'] = 'nBid'
       
    fdf['n3'] = imbalanceMsg.nsdq_AP
    Features['n3'] = 'nAsk'
    
    fdf['2_2'] = imbalanceMsg.Bid2
    Features['2_2'] = 'Bid2'
       
    fdf['3_2'] = imbalanceMsg.Ask2
    Features['3_2'] = 'Ask2'  
    
    fdf['9'] = imbalanceMsg.ImbRef-0.05
    Features['9'] = 'Ref-1'
    
    fdf['10'] = imbalanceMsg.ImbRef+0.05
    Features['10'] = 'Ref+1'
    
    fdf['11'] = imbalanceMsg.PrevCLC_P-0.05
    Features['11'] = 'Close-1'
    
    fdf['12'] = imbalanceMsg.PrevCLC_P+0.05
    Features['12'] = 'Close+1'
    
    fdf['13'] = imbalanceMsg.Bid_P-0.05
    Features['13'] = 'Bid-1'
    
    fdf['14'] = imbalanceMsg.Bid_P+0.05
    Features['14'] = 'Bid+1'
    
    fdf['15'] = imbalanceMsg.Ask_P-0.05
    Features['15'] = 'Ask-1'
    
    fdf['16'] = imbalanceMsg.Ask_P+0.05
    Features['16'] = 'Ask+1'
    
    fdf['17'] = imbalanceMsg.ImbRef+(imbalanceMsg.Ask_P - imbalanceMsg.Bid_P)
    Features['17'] = 'Ref+Spread'
    
    fdf['18'] = imbalanceMsg.ImbRef-(imbalanceMsg.Ask_P - imbalanceMsg.Bid_P)
    Features['18'] = 'Ref-Spread'
    
    fdf['19'] = imbalanceMsg.ImbRef-0.02
    Features['19'] = 'Ref-1'
    
    fdf['20'] = imbalanceMsg.ImbRef+0.02
    Features['20'] = 'Ref+1'
    
    fdf['21'] = imbalanceMsg.PrevCLC_P-0.02
    Features['21'] = 'Close-1'
    
    fdf['22'] = imbalanceMsg.PrevCLC_P+0.02
    Features['22'] = 'Close+1'
    
    fdf['23'] = imbalanceMsg.Bid_P-0.02
    Features['23'] = 'Bid-1'
    
    fdf['24'] = imbalanceMsg.Bid_P+0.02
    Features['24'] = 'Bid+1'
    
    fdf['25'] = imbalanceMsg.Ask_P-0.02
    Features['25'] = 'Ask-1'
    
    fdf['26'] = imbalanceMsg.Ask_P+0.02
    Features['26'] = 'Ask+1'
    
    return fdf, Features

# <codecell>

def getImbTime(nImb):
    startTime = '9:27:58'
    endTime = '9:28:03'
    if nImb==2:
        startTime = '9:28:03'
        endTime = '9:28:08'
    elif nImb==3:
        startTime = '9:28:08'
        endTime = '9:28:12'
    elif nImb==4:
        startTime = '9:28:12'
        endTime = '9:28:18'
    elif nImb==5:
        startTime = '9:28:18'
        endTime = '9:28:22'
    elif nImb==6:
        startTime = '9:28:22'
        endTime = '9:28:28'
    elif nImb==7:
        startTime = '9:28:28'
        endTime = '9:28:32'
    elif nImb==8:
        startTime = '9:28:32'
        endTime = '9:28:38'
    elif nImb==9:
        startTime = '9:28:38'
        endTime = '9:28:42'
    elif nImb==10:
        startTime = '9:28:42'
        endTime = '9:28:48'
    elif nImb==11:
        startTime = '9:28:48'
        endTime = '9:28:52'
    elif nImb==12:
        startTime = '9:28:52'
        endTime = '9:28:58'
    elif nImb==13:
        startTime = '9:28:58'
        endTime = '9:29:02'
    elif nImb==14:
        startTime = '9:29:02'
        endTime = '9:29:08'
    elif nImb==15:
        startTime = '9:29:08'
        endTime = '9:29:12'
    elif nImb==16:
        startTime = '9:29:12'
        endTime = '9:29:18'
    elif nImb==17:
        startTime = '9:29:18'
        endTime = '9:29:22'
    elif nImb==18:
        startTime = '9:29:22'
        endTime = '9:29:28'
    elif nImb==19:
        startTime = '9:29:28'
        endTime = '9:29:32'
    elif nImb==20:
        startTime = '9:29:32'
        endTime = '9:29:38'
    elif nImb==21:
        startTime = '9:29:38'
        endTime = '9:29:42'
    elif nImb==22:
        startTime = '9:29:42'
        endTime = '9:29:48'
    elif nImb==23:
        startTime = '9:29:48'
        endTime = '9:29:52'
    elif nImb==24:
        startTime = '9:29:52'
        endTime = '9:29:58'
    return (startTime,endTime)

def get_imbalanceMSG2(df,nImb):
    (startTime,endTime) = getImbTime(nImb)
    
    imbalanceMsg = df[df.Reason == 'Imbalance']
    imbalanceMsg = imbalanceMsg[
    (imbalanceMsg.ImbRef>0) & 
    (imbalanceMsg.ImbCBC>0) &
    (imbalanceMsg.ImbFar>0) &
    (imbalanceMsg.Ask_P>0) &
    (imbalanceMsg.Bid_P>0) #&
    #(imbalanceMsg.ImbShares!=0)
    ]
    imbalanceMsg.index = range(imbalanceMsg.shape[0])

    Timestamp = []
    for i in range(imbalanceMsg.shape[0]):
        Timestamp.append(datetime.datetime.strptime(imbalanceMsg.Time[i],'%H:%M:%S'))

    imbalanceMsg['Timestamp'] = Timestamp
    del Timestamp
    imbalanceMsg = imbalanceMsg.set_index(['Timestamp'])
    imbalanceMsg = imbalanceMsg.between_time(startTime,endTime)
    
    imbalanceMsg = imbalanceMsg[['Date','Symbol','Bid_P','Bid_S','Ask_P','Ask_S','ImbRef','ImbCBC','ImbFar','ImbShares','ImbPaired']]
    imbalanceMsg['Timestamp'] = imbalanceMsg.index
         
    #Getting additional info about previous day
    OPC = df[df.Reason == 'OPG']
    OPC = OPC[['Date','Symbol','tPrice']]
    OPC.columns = ['Date','Symbol','OPC_P']
    
    prev_CLC = df[df.tSide == 'YDAY']
    prev_CLC = prev_CLC[['Date','Symbol','tPrice','tShares']]
    prev_CLC.columns = ['Date','Symbol','PrevCLC_P','PrevCLC_S']
    
    #Adding OPC
    imbalanceMsg = pd.merge(imbalanceMsg, OPC, on=['Symbol','Date'])
    imbalanceMsg = pd.merge(imbalanceMsg, prev_CLC, on=['Symbol','Date'])

    #Filtering data with no prev OPC
    imbalanceMsg = imbalanceMsg[imbalanceMsg.OPC_P>0]
    imbalanceMsg.index = range(imbalanceMsg.shape[0])    
    
    #Adding new feature which reflects price move direction
    imbalanceMsg['Move'] = imbalanceMsg.Bid_P
    imbalanceMsg.Move = 0
    imbalanceMsg.Move[imbalanceMsg.OPC_P>imbalanceMsg.Ask_P] = 1
    imbalanceMsg.Move[imbalanceMsg.OPC_P<imbalanceMsg.Bid_P] = -1
    imbalanceMsg.Bid_S = imbalanceMsg.Bid_S.astype(float)
    imbalanceMsg.PrevCLC_P = imbalanceMsg.PrevCLC_P.astype(float)
       
    return imbalanceMsg   

# <codecell>

def get_imbelanceMSG3(df,nImb):
    (startTime,endTime) = getImbTime(nImb)
    
    imbalanceMsg = df[df.Reason == 'Imbalance']
    imbalanceMsg = imbalanceMsg[
    (imbalanceMsg.ImbRef>0) & 
    (imbalanceMsg.ImbCBC>0) &
    (imbalanceMsg.ImbFar>0) &
    (imbalanceMsg.ImbShares!=0)
    ]
    imbalanceMsg.index = range(imbalanceMsg.shape[0])
    
    Timestamp = []
    for i in range(imbalanceMsg.shape[0]):
        Timestamp.append(datetime.datetime.strptime(imbalanceMsg.Time[i],'%H:%M:%S'))

    imbalanceMsg['Timestamp'] = Timestamp
    del Timestamp
    imbalanceMsg = imbalanceMsg.set_index(['Timestamp'])
    imbalanceMsg = imbalanceMsg.between_time(startTime,endTime)

    imbalanceMsg = imbalanceMsg[['Date','Symbol','Bid_P','Bid_S','Ask_P','Ask_S',
                             'ImbRef','ImbCBC','ImbFar','ImbShares','ImbPaired',
                             'nsdq_BP', 'nsdq_BS', 'nsdq_AP', 'nsdq_AS','Bid2','Ask2']]
    imbalanceMsg['Timestamp'] = imbalanceMsg.index
         
    #Getting additional info about previous day
    OPC = df[df.tType == 'OPG']
    OPC = OPC[['Date','Symbol','tPrice']]
    OPC.columns = ['Date','Symbol','OPC_P']
    
    prev_CLC = df[df.tType == 'YDAY']
    prev_CLC = prev_CLC[['Date','Symbol','tVenue','tPrice']]
    prev_CLC.columns = ['Date','Symbol','PrevCLC_P','PrevCLC_S']
    
    #Adding OPC
    imbalanceMsg = pd.merge(imbalanceMsg, OPC, on=['Symbol','Date'])
    imbalanceMsg = pd.merge(imbalanceMsg, prev_CLC, on=['Symbol','Date'])

    #Filtering data with no prev OPC
    imbalanceMsg = imbalanceMsg[imbalanceMsg.OPC_P>0]
    imbalanceMsg.index = range(imbalanceMsg.shape[0])    
    
    #Adding new feature which reflects price move direction
    imbalanceMsg['Move'] = imbalanceMsg.Bid_P
    imbalanceMsg.Move = 0
    imbalanceMsg.Move[imbalanceMsg.OPC_P>imbalanceMsg.Ask_P] = 1
    imbalanceMsg.Move[imbalanceMsg.OPC_P<imbalanceMsg.Bid_P] = -1
    imbalanceMsg.Bid_S = imbalanceMsg.Bid_S.astype(float)
    imbalanceMsg.PrevCLC_P = imbalanceMsg.PrevCLC_P.astype(float)
       
    return imbalanceMsg   

# <codecell>

def Precision_Recall(cm, labels):
    m = cm.shape[0]
    sums1 = cm.sum(axis=1);
    sums2 = cm.sum(axis=0);
    precision = 0
    s1 = 0
    s2 = 0
    for i in range(m):
        if labels[i] == 0: continue;
        precision +=  cm[i,i]
        s1 += sums1[i]
        s2 += sums2[i]

    return precision/s2, precision/s1, 2*(precision/s1 * precision/s2)/(precision/s2 + precision/s1)

# <codecell>

def run_cv_proba(X,y,clf_class,n_folds,test_size,dates,datesDF,**kwargs):
    import pandas
    import numpy
    from sklearn.metrics import confusion_matrix
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    from sklearn.ensemble import RandomForestClassifier as RF
    import numpy
    
    labels =  numpy.sort(list(set(y)))
    test_cm = np.zeros((len(labels),len(labels)))
    train_cm = np.zeros((len(labels),len(labels)))
    
    CLF_BEST = clf_class()
    TEST_F_Score = 0
    
    for i in range(n_folds): 
        #======Get test_train_split=============
        r = range(len(dates))
        np.random.shuffle(r)
        test_days = r[:test_size] 
        train_days = r[test_size:] 

        Xtrain = X.ix[datesDF.ix[train_days],:]
        Xtest = X.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]] 
        
        Xtrain.index = range(Xtrain.shape[0])
        Xtest.index = range(Xtest.shape[0])
        ytrain.index = range(ytrain.shape[0])
        ytest.index = range(ytest.shape[0])
        #========================================        
            
        if clf_class=='NN':    
            from pybrain.tools.shortcuts import buildNetwork
            from pybrain.datasets import SupervisedDataSet
            from pybrain.datasets import ClassificationDataSet
            from pybrain.structure.modules   import SoftmaxLayer

            #Create dataset
            #ds = SupervisedDataSet(Xtrain.shape[1], 1)
            ds = ClassificationDataSet(Xtrain.shape[1], 1, nb_classes=2)
            for j in range(Xtrain.shape[0]):
                ds.addSample(Xtrain.ix[j,:], ytrain.ix[j,:])
            ds._convertToOneOfMany( )

            #Create net
            #net = buildNetwork(ds.indim, ds.indim*2, ds.outdim, outclass=SoftmaxLayer)

            #varsion1
            #from pybrain.structure import FeedForwardNetwork
            #net = FeedForwardNetwork()
            from pybrain.structure import LinearLayer, SigmoidLayer
            #inLayer = LinearLayer(ds.indim)
            #hiddenLayer = SigmoidLayer(ds.indim)
            #outLayer = SoftmaxLayer(ds.outdim)
            #net.addInputModule(inLayer)
            #net.addModule(hiddenLayer)
            #net.addOutputModule(outLayer)
            from pybrain.structure import FullConnection
            #in_to_hidden = FullConnection(inLayer, hiddenLayer)
            #hidden_to_out = FullConnection(hiddenLayer, outLayer)
            #net.addConnection(in_to_hidden)
            #net.addConnection(hidden_to_out)
            #net.sortModules()
            
            #varsion2
            from pybrain.structure import RecurrentNetwork
            net = RecurrentNetwork()
            net.addInputModule(LinearLayer(ds.indim, name='inLayer'))
            net.addModule(SigmoidLayer(ds.indim, name='hiddenLayer'))
            net.addOutputModule(SoftmaxLayer(ds.outdim, name='outLayer'))
            net.addConnection(FullConnection(net['inLayer'], net['hiddenLayer'], name='in_to_hidden'))
            net.addConnection(FullConnection(net['hiddenLayer'], net['outLayer'], name='hidden_to_out'))
            net.addRecurrentConnection(FullConnection(net['hiddenLayer'], net['hiddenLayer'], name='hidden_to_hidden'))
            net.sortModules()

            #Train net
            from pybrain.supervised.trainers import BackpropTrainer
            trainer = BackpropTrainer(net, ds, momentum=0.1, verbose=True, weightdecay=0.01)
            trainer.train()
            #trainer.trainUntilConvergence(dataset=ds,maxEpochs=10)
            
            if False:#combination of NN and COMB
                #get new features
                Xtrain_new= numpy.zeros((Xtrain.shape[0],hiddenLayer.dim),float)
                Xtest_new= numpy.zeros((Xtest.shape[0],hiddenLayer.dim),float)

                for j in range(Xtrain.shape[0]):
                    to_hidden=numpy.dot(in_to_hidden.params.reshape(hiddenLayer.dim,inLayer.dim),\
                                        Xtrain.ix[j,:].as_matrix())
                    Xtrain_new[j,:] = hiddenLayer.activate(to_hidden)
                for j in range(Xtest.shape[0]):
                    to_hidden=numpy.dot(in_to_hidden.params.reshape(hiddenLayer.dim,inLayer.dim),\
                                        Xtest.ix[j,:].as_matrix())
                    Xtest_new[j,:] = hiddenLayer.activate(to_hidden)

                #Work with new features
                clf1 = RF(n_jobs=2,min_samples_split = Xtrain.shape[0]*0.05, criterion = 'entropy')
                clf2 = GBC(init='zero')

                clf1.fit(Xtrain_new,ytrain)
                clf2.fit(Xtrain_new,ytrain)

                probaTest1=clf1.predict_proba(Xtest_new).astype(float)
                probaTest2=clf2.predict_proba(Xtest_new).astype(float)
                for i in range(probaTest1.shape[0]):
                    for j in range(probaTest1.shape[1]):
                         probaTest1[i,j]=0.5*(probaTest1[i,j]+probaTest2[i,j])


                probaTrain1=clf1.predict_proba(Xtrain_new).astype(float)
                probaTrain2=clf2.predict_proba(Xtrain_new).astype(float)
                for i in range(probaTrain1.shape[0]):
                    for j in range(probaTrain1.shape[1]):
                        probaTrain1[i,j]=0.5*(probaTrain1[i,j]+probaTrain2[i,j])

                ypred = clf1.classes_[numpy.argmax(probaTest1,axis=1)]
                ypredTrain = clf1.classes_[numpy.argmax(probaTrain1,axis=1)] 
            else:
                ypred = ytest.copy()
                ypredTrain = ytrain.copy()

                for j in range(Xtrain.shape[0]):
                    ypredTrain.ix[j]=net.activate(Xtrain.ix[j,:])[1]>0.5
                for j in range(Xtest.shape[0]):
                    ypred.ix[j]=net.activate(Xtest.ix[j,:])[1]>0.5
                
            test_cm += confusion_matrix(ytest.astype(bool),ypred.astype(bool),labels).astype(float)/n_folds
            train_cm += confusion_matrix(ytrain.astype(bool),ypredTrain.astype(bool),labels).astype(float)/n_folds
             
            continue;    
            
        if clf_class=='B':
            ypred = ytest.copy(); ypred[:] = 0
            ypredTrain = ytrain.copy(); ypredTrain[:] = 0
            if (any(Xtest.columns=='D4')):
                ypred[(Xtest.D4>=0)] = 1
                ypredTrain[(Xtrain.D4>=0)] = 1
            if (any(Xtest.columns=='D5')):
                ypred[(Xtest.D5>=0)] = 1
                ypredTrain[(Xtrain.D5>=0)] = 1
            
            continue;        
            
        if (clf_class=='COMB'):
                clf1 = RF(n_jobs=2,min_samples_split = Xtrain.shape[0]*0.05, criterion = 'entropy')
                clf2 = GBC(min_samples_split = Xtrain.shape[0]*0.05,init='zero')# learning_rate=0.1
                
                #z = float(len(ytrain[ytrain==0]))
                #nall = float(len(ytrain))
                #sample_weight = np.array([(nall-z)/nall if i == 0 else z/nall for i in ytrain])
                clf1.fit(Xtrain,ytrain)
                clf2.fit(Xtrain,ytrain)
                
                probaTest1=clf1.predict_proba(Xtest).astype(float)
                probaTest2=clf2.predict_proba(Xtest).astype(float)
                for i in range(probaTest1.shape[0]):
                    for j in range(probaTest1.shape[1]):
                        probaTest1[i,j]=0.5*(probaTest1[i,j]+probaTest2[i,j])

                        
                probaTrain1=clf1.predict_proba(Xtrain).astype(float)
                probaTrain2=clf2.predict_proba(Xtrain).astype(float)
                for i in range(probaTrain1.shape[0]):
                    for j in range(probaTrain1.shape[1]):
                        probaTrain1[i,j]=0.5*(probaTrain1[i,j]+probaTrain2[i,j])
                                 
                ypred = clf1.classes_[numpy.argmax(probaTest1,axis=1)]
                ypredTrain = clf1.classes_[numpy.argmax(probaTrain1,axis=1)]  

                test_cm += confusion_matrix(ytest,ypred,labels).astype(float)/n_folds
                train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds
            
                continue;
            
        else: 
                if (type(clf_class()) ==  type(LR())) :
                    clf = clf_class(class_weight='auto',C=0.1)
                if (type(clf_class()) ==  type(SVC())):
                    clf = clf_class(class_weight='auto',probability=True)
                if (type(clf_class()) ==  type(RF())):
                    #print 1
                    clf = clf_class(n_jobs=4,min_samples_split = Xtrain.shape[0]*0.05, \
                                   criterion = 'entropy', n_estimators = 10)# min_samples_leaf = Xtrain.shape[0]*0.05,
                if (type(clf_class()) ==  type(GBC())):
                    clf = clf_class(min_samples_split = Xtrain.shape[0]*0.05,init='zero')#min_samples_split = Xtrain.shape[0]*0.05
                if (type(clf_class()) ==  type(GBR())):
                    clf = clf_class(init='zero')

                clf.fit(Xtrain,ytrain)
                #print type(clf)
            
                if (type(clf_class()) !=  type(GBR())):
                    probaTest = clf.predict_proba(Xtest).astype(float)
                    probaTrain = clf.predict_proba(Xtrain).astype(float)

                    ypred = clf.classes_[numpy.argmax(probaTest,axis=1)]
                    ypredTrain = clf.classes_[numpy.argmax(probaTrain,axis=1)]  

                    test_cm_tmp = confusion_matrix(ytest,ypred,labels).astype(float)
                    test_cm += test_cm_tmp/n_folds
                    train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds
                    
                else:
                    probaTest = clf.predict(Xtest).astype(float)
                    probaTrain = clf.predict(Xtrain).astype(float)

                    ypred = probaTest>0.5
                    ypredTrain = probaTrain>0.5  

                    test_cm_tmp = confusion_matrix(ytest,ypred,labels).astype(float)
                    test_cm += test_cm_tmp/n_folds
                    train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds 
                    
                pr =  Precision_Recall(test_cm_tmp, labels)
                #print pr
                if (pr[2]>TEST_F_Score):
                     TEST_F_Score =  pr[2]
                     CLF_BEST = clf  
    
    
    #print TEST_ERRORS
    print "max F_Score ",TEST_F_Score
    
    #print CLF_BEST

    test_pr = Precision_Recall(test_cm, labels)
    train_pr = Precision_Recall(train_cm, labels)    
    return 1-train_pr[2], 1-test_pr[2], test_cm, CLF_BEST, TEST_F_Score

# <codecell>

def dates_tmp_df(fdf):
    import numpy
    
    datesDF = pd.DataFrame()
    datesDF['Date'] = fdf.Date
    datesDF['newIndex'] = numpy.zeros((datesDF.shape[0],1))
    datesDF.index = range(datesDF.shape[0])

    dates = sorted(list(set(fdf.Date)))
    for i in range(datesDF.shape[0]):
        for j in range(len(dates)):
            if (datesDF.Date[i]==dates[j]):            
                datesDF.newIndex[i] = j
            
    datesDF.index = datesDF.newIndex 
    datesDF.newIndex = range(datesDF.shape[0])
    datesDF = datesDF['newIndex']
    
    return datesDF

def run_cv2(X,y,clf_class,n_folds,test_size,dates,datesDF,**kwargs):
    from sklearn.metrics import confusion_matrix
    from sklearn.cross_validation import train_test_split
    from sklearn import preprocessing
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.ensemble import RandomForestRegressor as RFR
    import numpy
    
    if (type(clf_class()) !=  type(RFR())):
        labels =  numpy.sort(list(set(y)))
    else:
        labels =  numpy.sort(list(set(np.sign(y))))
        
    test_cm = np.zeros((len(labels),len(labels)))
    train_cm = np.zeros((len(labels),len(labels)))
    testError = 0
    trainError = 0
    
    for i in range(n_folds):  
        r = range(len(dates))
        np.random.shuffle(r)
        test_days = r[:test_size] 
        train_days = r[test_size:] 
               
        Xtrain = X.ix[datesDF.ix[train_days],:]
        Xtest = X.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]]
        
        if clf_class!='B':
            if not kwargs.has_key('n_ensembles'):
                n_ensembles = 1
                test_size_ensemble = 0
            else:
                n_ensembles = kwargs['n_ensembles']
                test_size_ensemble = kwargs['test_size_ensemble']
                
            ypred = ytest.copy(); ypred[:] = 0
            ypredTrain = ytrain.copy(); ypredTrain[:] = 0
            for j in range(n_ensembles):  
                Xtrain_sub, Xtest_sub, ytrain_sub, ytest_sub = train_test_split(Xtrain, ytrain, test_size=test_size_ensemble)

                if (type(clf_class()) ==  type(LR())) | (type(clf_class()) ==  type(SVC())):
                    clf = clf_class(class_weight='auto')
                else:
                    clf = clf_class(n_jobs=2,min_samples_split = Xtrain.shape[0]*0.05)
                    
                clf.fit(Xtrain_sub,ytrain_sub)

                ypred += clf.predict(Xtest).astype(float)/n_ensembles
                ypredTrain += clf.predict(Xtrain).astype(float)/n_ensembles

            #Averaging of assemblies results
            if (type(clf_class()) !=  type(RFR())):
                ypred[ypred>0.5] = 1
                ypred[ypred<-0.5] = -1
                ypred[(ypred!=1) & (ypred!=-1)] = 0

                ypredTrain[ypredTrain>0.5] = 1
                ypredTrain[ypredTrain<-0.5] = -1
                ypredTrain[(ypredTrain!=1) & (ypredTrain!=-1)] = 0
        else:
            ypred = ytest.copy(); ypred[:] = 0
            ypredTrain = ytrain.copy(); ypredTrain[:] = 0
            ypred[(Xtest.a14>0) & (Xtest.D5>=0)] = 1
            ypred[(Xtest.a14<0) & (Xtest.D4>=0)] = -1
            ypredTrain[(Xtrain.a14>0) & (Xtrain.D5>=0)] = 1
            ypredTrain[(Xtrain.a14<0) & (Xtrain.D4>=0)] = -1
        
        if (clf_class=='B'):
            test_cm += confusion_matrix(ytest,ypred,labels).astype(float)/n_folds
            train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds
        else:
            if (type(clf_class()) ==  type(RFR())):
                plt.scatter(ytest,ypred)
                test_cm += confusion_matrix(np.sign(ytest),np.sign(ypred),labels).astype(float)/n_folds
                train_cm += confusion_matrix(np.sign(ytrain),np.sign(ypredTrain),labels).astype(float)/n_folds
            else:
                test_cm += confusion_matrix(ytest,ypred,labels).astype(float)/n_folds
                train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds
        
    test_pr = Precision_Recall(test_cm, labels)
    train_pr = Precision_Recall(train_cm, labels)    
    return 1-train_pr[2], 1-test_pr[2], test_cm

def OneModelResults(clf_class, input,target,ERRORS,dates,datesDF,**kwargs):
    import numpy
    fig1 = plt.figure(figsize=(15, 5))
    plt.clf()
    ax1 = fig1.add_subplot(1,3,1)
    trainError, testError, cm, clf, fscore = run_cv_proba(input,target,clf_class,30,1,dates,datesDF,**kwargs)
    draw_confusion_matrix(cm,  numpy.sort(list(set(target))), fig1, ax1)
    ERRORS.loc[ERRORS.shape[0]] =[str(clf_class).split('.')[-1].strip('>'),trainError,testError]
    pr = Precision_Recall(cm,numpy.sort(list(set(target))))
    cm2return=cm
    print 'Precision - %s, Recall - %s, F_Score - %s' % (pr[0],pr[1],pr[2])

    #if clf_class=='NN':
    return cm2return
    
    #Show learning curves
    TrainError=[]
    TestError=[]
    nDays = len(dates)
    testRange = range(nDays-1)
    for i in testRange: 
        trainError, testError, cm, clf, fscore = run_cv_proba(input,target,clf_class,5,i+1,dates,datesDF,**kwargs)
        #print i,testError
        TrainError.append(trainError)
        TestError.append(testError)

    LearningCurves = pd.DataFrame()
    LearningCurves['Index'] = testRange
    LearningCurves['Index']+= 1
    LearningCurves['TrainError'] = TrainError
    LearningCurves['TestError'] = TestError
    LearningCurves['Index'] = nDays-LearningCurves['Index']
    LearningCurves = pd.melt(LearningCurves, id_vars = 'Index', value_vars = ['TestError','TrainError'])

    g = ggplot(LearningCurves, aes('Index', 'value', color = 'variable')) + geom_step() + \
    ggtitle('Learning curves') + xlab("% of data sent to train") + ylab("Error")
    
    return g,cm2return

def Regression(clf_class, input,target,dates,datesDF,side):
    import numpy
    trainError, testError = run_reg(input,target,clf_class,5,1,dates,datesDF,side)
    print 'TrainError - %s, TestError - %s' % (trainError, testError)
    #Show learning curves
    TrainError=[]
    TestError=[]
    nDays = len(dates)
    testRange = range(5,nDays/2-1)
    for i in testRange: 
        trainError, testError = run_reg(input,target,clf_class,5,i+1,dates,datesDF,side)
        TrainError.append(trainError)
        TestError.append(testError)

    LearningCurves = pd.DataFrame()
    LearningCurves['Index'] = testRange
    LearningCurves['Index']+= 1
    LearningCurves['TrainError'] = TrainError
    LearningCurves['TestError'] = TestError
    LearningCurves['Index'] = nDays-LearningCurves['Index']
    LearningCurves = pd.melt(LearningCurves, id_vars = 'Index', value_vars = ['TestError','TrainError'])

    g = ggplot(LearningCurves, aes('Index', 'value', color = 'variable')) + geom_step() + \
    ggtitle('Learning curves') + xlab("% of data sent to train") + ylab("Error")
    
    return g

def GetNEstimators(clf_class, input,target,dates,datesDF,side):
    TrainError=[]
    TestError=[]
    nDays = len(dates)
    testRange = range(90)
    
    for i in testRange: 
        trainError, testError = run_reg_2(input,target,clf_class,1,nDays/10,dates,datesDF,side,i)
        TrainError.append(trainError)
        TestError.append(testError)

    LearningCurves = pd.DataFrame()
    LearningCurves['Index'] = testRange
    LearningCurves['Index']+= 1
    LearningCurves['TrainError'] = TrainError
    LearningCurves['TestError'] = TestError
    LearningCurves['Index'] = testRange
    LearningCurves['Index'] = 100+LearningCurves['Index']*10
    LearningCurves = pd.melt(LearningCurves, id_vars = 'Index', value_vars = ['TestError','TrainError'])

    g = ggplot(LearningCurves, aes('Index', 'value', color = 'variable')) + geom_step() + \
    ggtitle('Learning curves') + xlab("% of data sent to train") + ylab("Error")
    
    return g

def run_reg_2(X,y,clf_class,n_folds,test_size,dates,datesDF,side,nest):
    import numpy
    from sklearn import metrics
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    from sklearn.ensemble import RandomForestRegressor as RFR
    from sklearn.neighbors import KNeighborsRegressor as KNR
    
    X.index = range(X.shape[0])
    y.index = range(y.shape[0])
    
    test_pr = 0.0;
    train_pr = 0.0;
    
    for i in range(n_folds): 
        r = range(len(dates))
        np.random.shuffle(r)
        test_days = r[:test_size] 
        train_days = r[test_size:] 

        Xtrain = X.ix[datesDF.ix[train_days],:]
        Xtest = X.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]]
            
        if side>0:
            f1 = ytrain>0
            f2 = ytest>0
        else:
            f1 = ytrain<0
            f2 = ytest<0
        
        clf = clf_class(loss = 'huber',n_estimators=100+nest*10)

        clf.fit(Xtrain[f1], ytrain[f1])

        ypred = clf.predict(Xtest[f2])
        ypredTrain = clf.predict(Xtrain[f1])

        test_pr += metrics.r2_score(ytest[f2], ypred)/n_folds
        train_pr += metrics.r2_score(ytrain[f1], ypredTrain)/n_folds
    
    return 1-train_pr, 1-test_pr

def run_reg(X,y,clf_class,n_folds,test_size,dates,datesDF,side):
    import numpy
    from sklearn import metrics
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    from sklearn.ensemble import RandomForestRegressor as RFR
    from sklearn.neighbors import KNeighborsRegressor as KNR
    
    X.index = range(X.shape[0])
    y.index = range(y.shape[0])
    
    test_pr = 0.0;
    train_pr = 0.0;
    
    for i in range(n_folds): 
        r = range(len(dates))
        np.random.shuffle(r)
        test_days = r[:test_size] 
        train_days = r[test_size:] 

        Xtrain = X.ix[datesDF.ix[train_days],:]
        Xtest = X.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]]
            
        if side>0:
            f1 = ytrain>0
            f2 = ytest>0
        else:
            f1 = ytrain<0
            f2 = ytest<0
        
        if (clf_class!='COMB'):
            if (type(clf_class()) ==  type(GBR())):
                clf = clf_class(loss = 'huber')#learning_rate=0.1,
            if (type(clf_class()) ==  type(KNR())):
                clf = clf_class(weights = 'uniform',n_neighbors=20)
            else:
                clf = clf_class()

            clf.fit(Xtrain[f1], ytrain[f1])

            ypred = clf.predict(Xtest[f2])
            ypredTrain = clf.predict(Xtrain[f1])
        else:
            clf1 = GBR(loss = 'huber',min_samples_split = ytrain[f1].shape[0]*0.05)
            clf2 = RFR(min_samples_split = ytrain[f1].shape[0]*0.05)
            clf1.fit(Xtrain[f1], ytrain[f1])
            clf2.fit(Xtrain[f1], ytrain[f1])
            
            ypred = 0.5*(clf1.predict(Xtest[f2])+clf2.predict(Xtest[f2]))
            ypredTrain =0.5*( clf1.predict(Xtrain[f1])+ clf2.predict(Xtrain[f1]))

        test_pr += metrics.r2_score(ytest[f2], ypred)/n_folds
        train_pr += metrics.r2_score(ytrain[f1], ypredTrain)/n_folds
    
    return 1-train_pr, 1-test_pr

# <codecell>

def run_cvNN(X,y,n_folds,test_size,**kwargs):
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn import preprocessing
    import neurolab as nl
    
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    
    labels =  sort(list(set(y)))
    test_cm = np.zeros((len(labels),len(labels)))
    train_cm = np.zeros((len(labels),len(labels)))
    testError = 0
    trainError = 0
    
    for i in range(n_folds):  
        r = range(len(dates))
        np.random.shuffle(r)
        test_days = r[:test_size] 
        train_days = r[test_size:] 
               
        Xtrain = X.ix[datesDF.ix[train_days],:]
        Xtest = X.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]]
        
        scaler = preprocessing.StandardScaler().fit(Xtrain)
        
        input = scaler.transform(Xtrain)  
        
        net = nl.net.newff(nl.tool.minmax(input), **kwargs)
        error = net.train(input, lb.transform(ytrain),show=500)
        
        ypred = lb.inverse_transform(net.sim(scaler.transform(Xtest)))
        ypredTrain = lb.inverse_transform(net.sim(input))
        
        ypred = clf.predict(Xtest)
        ypredTrain = clf.predict(Xtrain)

        test_cm += confusion_matrix(ytest,ypred,labels).astype(float)/n_folds
        train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds
        
    test_pr = Precision_Recall(test_cm)
    train_pr = Precision_Recall(train_cm)    
    return 1-train_pr[2], 1-test_pr[2], test_cm


def run_cvNN2(X,y,n_folds,threshold,test_size,**kwargs):
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn import preprocessing
    import neurolab as nl
    
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    
    labels =  sort(list(set(y)))
    test_cm = np.zeros((len(labels),len(labels)))
    train_cm = np.zeros((len(labels),len(labels)))
    testError = 0
    trainError = 0
    
    for i in range(n_folds):  
        r = range(len(dates))
        np.random.shuffle(r)
        test_days = r[:test_size] 
        train_days = r[test_size:] 
               
        Xtrain = X.ix[datesDF.ix[train_days],:]
        Xtest = X.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]]
        
        scaler = preprocessing.StandardScaler().fit(Xtrain)
        
        input = scaler.transform(Xtrain)  
        
        net = nl.net.newff(nl.tool.minmax(input), **kwargs)
        error = net.train(input, ytrain.reshape(len(ytrain),1),show=500)
        
        ypred = net.sim(scaler.transform(Xtest)).flatten()
        ypred[ypred>threshold] = 1
        ypred[ypred<-threshold] = -1
        ypred[(ypred!=1) & (ypred!=-1)] = 0
        
        ypredTrain = net.sim(input).flatten()
        ypredTrain[ypredTrain>threshold] = 1
        ypredTrain[ypredTrain<-threshold] = -1
        ypredTrain[(ypredTrain!=1) & (ypredTrain!=-1)] = 0
        
        test_cm += confusion_matrix(ytest,ypred,labels).astype(float)/n_folds
        train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds
        
    test_pr = Precision_Recall(test_cm)
    train_pr = Precision_Recall(train_cm)    
    return 1-train_pr[2], 1-test_pr[2], test_cm


# <codecell>

def draw_confusion_matrix(conf_arr, labels, fig, ax):  
    #print conf_arr
    conf_arr=conf_arr.astype(float)
    
    sums = conf_arr.sum(axis=0)
    #print sums
    for i in range(len(labels)):
        conf_arr[:,i] /= sums[i]
    #print conf_arr
    #fig = plt.figure()
    #plt.clf()
    #ax = fig.add_subplot(111)
    #ax.set_aspect(1)
    res = ax.imshow(np.array(conf_arr), cmap=plt.cm.jet, interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y])[:4], xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    #cb = fig.colorbar(res)
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])
    #plt.show()

# <codecell>

def get_signals1(imbalanceMsg,X,y,clf_class,dates,datesDF,**kwargs):  
    import numpy
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression as LR
    labels = numpy.sort(list(set(y)))
    
    Signals = imbalanceMsg[['Date','Timestamp','Symbol','Ask_P','Bid_P']]
    Signals['Side'] = numpy.zeros((Signals.shape[0],1))
    Signals['Price'] = Signals.Ask_P   
    
    
    if clf_class!='B':
        for i in range(int(datesDF.index.max())+1):  
            train_days = range(len(dates))
            test_days = i 
            train_days.remove(i)

            Xtrain = X.ix[datesDF.ix[train_days],:]
            Xtest = X.ix[datesDF.ix[test_days],:]
            ytrain = y.ix[datesDF.ix[train_days]]
            ytest = y.ix[datesDF.ix[test_days]]
               
            if not kwargs.has_key('n_ensembles'):
                
                if (type(clf_class()) ==  type(LR())) | (type(clf_class()) ==  type(SVC())):
                    clf = clf_class(class_weight='auto')
                else:
                    clf = clf_class(n_jobs=2,min_samples_split = Xtrain.shape[0]*0.05)
                
                clf.fit(Xtrain,ytrain)

                Signals.Side[datesDF.ix[test_days]] = clf.predict(Xtest)
            else:
                n_ensembles = kwargs['n_ensembles']
                test_size_ensemble = kwargs['test_size_ensemble']
                
                ypred = ytest.copy(); ypred[:] = 0
                for j in range(n_ensembles):  
                    Xtrain_sub, Xtest_sub, ytrain_sub, ytest_sub = train_test_split(Xtrain, ytrain, test_size=test_size_ensemble)

                    if (type(clf_class()) ==  type(LR())) | (type(clf_class()) ==  type(SVC())):
                        clf = clf_class(class_weight='auto')
                    else:
                        clf = clf_class(n_jobs=2,min_samples_split = Xtrain.shape[0]*0.05)
                        
                    clf.fit(Xtrain_sub,ytrain_sub)
                    
                    ypred += clf.predict(Xtest).astype(float)/n_ensembles


                #Averaging of assemblies results
                ypred[ypred>0.5] = 1
                ypred[ypred<-0.5] = -1
                ypred[(ypred!=1) & (ypred!=-1)] = 0
                

                Signals.Side[datesDF.ix[test_days]] = ypred
            
            Signals.Side[(Signals.Side!=0) & (X.a14>0)] = 1
            Signals.Side[(Signals.Side!=0) & (X.a14<0)] = -1
    else:
        ypred = y.copy(); ypred[:] = 0
        ypred[(X.a14>0) & (X.D5>=0)] = 1
        ypred[(X.a14<0) & (X.D4>=0)] = -1
        Signals.Side = ypred
        
        
    Signals.Price[Signals['Side']==1] = Signals.Ask_P[Signals['Side']==1]
    Signals.Price[Signals['Side']==-1] = Signals.Bid_P[Signals['Side']==-1]
    Signals = Signals[Signals.Side!=0]
    Signals = Signals[['Date','Timestamp','Symbol','Price','Side']] 
    Signals.index = Signals.Timestamp
    return Signals

def get_signals_clf(imbalanceMsg,X,y,clf,dates,datesDF,**kwargs):  
    import numpy
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression as LR
    labels = numpy.sort(list(set(y)))
    
    Signals = imbalanceMsg[['Date','Timestamp','Symbol','Ask_P','Bid_P']]
    Signals['Side'] = numpy.zeros((Signals.shape[0],1))
    Signals['Price'] = Signals.Ask_P   
                       
    ypred = clf.predict(X).astype(float)

    ypred[ypred>0.5] = 1
    ypred[ypred<-0.5] = -1
    ypred[(ypred!=1) & (ypred!=-1)] = 0

    Signals.Side = ypred      
       
    Signals.Price[Signals['Side']==1] = Signals.Ask_P[Signals['Side']==1]
    Signals.Price[Signals['Side']==-1] = Signals.Bid_P[Signals['Side']==-1]
    Signals = Signals[Signals.Side!=0]
    Signals = Signals[['Date','Timestamp','Symbol','Price','Side']] 
    Signals.index = Signals.Timestamp
    return Signals

def get_signals_proba(imbalanceMsg,X,y,clf_class,dates,datesDF,**kwargs):  
    import numpy
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression as LR
    labels = numpy.sort(list(set(y)))
    
    Signals = imbalanceMsg[['Date','Timestamp','Symbol','Ask_P','Bid_P']]
    Signals['Side'] = numpy.zeros((Signals.shape[0],1))
    Signals['Price'] = Signals.Ask_P   
    
    for i in range(int(datesDF.index.max())+1):  
        train_days = range(len(dates))
        test_days = i 
        train_days.remove(i)
               
        Xtrain = X.ix[datesDF.ix[train_days],:]
        Xtest = X.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]]
                       
        if (type(clf_class()) ==  type(LR())) | (type(clf_class()) ==  type(SVC())):
            clf = clf_class(class_weight='auto')
        else:
            clf = clf_class(n_jobs=2,min_samples_split = Xtrain.shape[0]*0.05)
                
        clf.fit(Xtrain,ytrain)

        Signals.Side[datesDF.ix[test_days]] = clf.classes_[numpy.argmax(clf.predict_proba(Xtest),axis=1)]
              
    #Signals.Side[Signals.Side==1] = 0
    #Signals.Side[Signals.Side==-1] = 0
    Signals.Side[Signals.Side>0] = 1
    Signals.Side[Signals.Side<0] = -1
    
    Signals.Price[Signals['Side']==1] = Signals.Ask_P[Signals['Side']==1]
    Signals.Price[Signals['Side']==-1] = Signals.Bid_P[Signals['Side']==-1]
    Signals = Signals[Signals.Side!=0]
    Signals = Signals[['Date','Timestamp','Symbol','Price','Side']] 
    Signals.index = Signals.Timestamp
    return Signals
        

# <codecell>

def get_performance(Signals,df,days,add=0):
    #days = sorted(list(set(Signals.index.date)))
    
    PNL=[]
    NEGPNL=[]
    SDict=dict()
    for day in days:
        dayPnl=[]
        negdayPnl=[]
        #print '----------------------------------'
        #print day
        #print '----------------------------------'
        curr_day_signals = Signals[Signals.Date==day]
        data = df[df.Date==day]
        buys = curr_day_signals[curr_day_signals.Side==1]
        buys = buys.sort(['Symbol'])
        sells = curr_day_signals[curr_day_signals.Side==-1]
        sells = sells.sort(['Symbol'])
        
        #endTimestamp = pd.Timestamp('09:30:01')
        #print 'BUY'
        for count, row in buys.iterrows():
            symbol = row[2]
            price = row[3]+add
            curr_symbol_data=data[data.Symbol==symbol]#.between_time(startTimestamp,endTimestamp)

            #AskPrices = dict()
            #for i in range(curr_symbol_data.shape[0]):
            #    if curr_symbol_data.Ask_P[i]>price: continue
            #    if (not AskPrices.has_key(curr_symbol_data.Ask_P[i])):
            #        AskPrices[curr_symbol_data.Ask_P[i]] = curr_symbol_data.Ask_S[i]
                    
            volumeTraded = 1#min(abs(curr_symbol_data.ImbShares[0]),curr_symbol_data.Ask_S[0]       
            OPC = curr_symbol_data.tPrice[(curr_symbol_data.tType=='OPG')]
            pnl = volumeTraded*(OPC.values[0]-price)    
            dayPnl.append(pnl)
            if pnl<0:
                negdayPnl.append(pnl)
            
            #if pnl<0:
            #    print ' BUY   %s %s shares at %s SELL at %s PNL %s' % ( symbol,volumeTraded,price,OPC.values[0],pnl)
             
        for count, row in sells.iterrows():
            symbol = row[2]
            price = row[3]-add
            curr_symbol_data=data[data.Symbol==symbol]#.between_time(startTimestamp,endTimestamp)
            
            #BidPrices = dict()
            #for i in range(curr_symbol_data.shape[0]):
            #    if curr_symbol_data.Bid_P[i]<price: continue
            #    if (not BidPrices.has_key(curr_symbol_data.Bid_P[i])):
            #        BidPrices[curr_symbol_data.Bid_P[i]] = curr_symbol_data.Bid_S[i]
            
            volumeTraded = 1#min(abs(curr_symbol_data.ImbShares[0]),curr_symbol_data.Bid_S[0])
            OPC = curr_symbol_data.tPrice[curr_symbol_data.tType=='OPG']
            pnl = volumeTraded*(price-OPC.values[0])
            dayPnl.append(pnl)
            if pnl<0:
                negdayPnl.append(pnl)
                
            #if pnl<0:
            #    print ' SELL  %s %s shares at %s BUY as %s PNL %s' % ( symbol,volumeTraded,price,OPC.values[0],pnl)        
            
        PNL.append(np.sum(dayPnl))
        NEGPNL.append(np.sum(negdayPnl))
        print '%s %s' % (day,np.sum(dayPnl))
      
    result = pd.DataFrame()
    result['Date'] = days
    result['Pnl'] = PNL
    print '%s %s' % (np.sum(NEGPNL),np.sum(PNL))
    return result

# <codecell>

def Tree2Txt(clf,t,fileName):
    f = open(fileName, 'w+')
    f.write(str(t.n_classes[0])+'\n');
    f.write(str(t.n_features)+'\n');
    f.write(str(t.capacity)+'\n');
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    if (type(clf) !=  type(GBR())):
        for i in range(t.capacity):
            s= '%d;%.19f;' % (t.feature[i],t.threshold[i])
            if t.n_classes[0]==1:
                s +='%s;' % str(t.value[i][0][0])
            else:
                for j in range(t.n_classes[0]):
                    Sum = sum(t.value[i][0])
                    if Sum>0:
                        s +='%s;' %  str(t.value[i][0][j]/Sum)
                    else:
                         s +='%s;' % '0'
            f.write(s+'\n')
    else:
        for i in range(t.capacity):
            s= '%d;%.19f;' % (t.feature[i],t.threshold[i])
            s +='%s;' % str(t.value[i][0][0]*clf.learning_rate*clf.n_estimators)
            f.write(s+'\n')
    f.close()
    
def Tree2Sql(clf,t,RFID,ID, datatypes,con):
    data = np.zeros((t.capacity+3,7))
    data[:,0] = RFID
    data[:,1] = ID
    dataID = datatypes[datatypes.Name=='data'].ID
    data[3:,2] = dataID
    
    data[0,2:] = [datatypes[datatypes.Name=='n_classes'].ID,t.n_classes[0],0,0,0]
    data[1,2:] = [datatypes[datatypes.Name=='n_features'].ID,t.n_features,0,0,0]
    data[2,2:] = [datatypes[datatypes.Name=='capacity'].ID,t.capacity,0,0,0]
    
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    if (type(clf) !=  type(GBR())):
        for i in range(t.capacity):
            data[i+3,3] = i
            data[i+3,4] = t.feature[i]
            data[i+3,5] = t.threshold[i]
            if t.n_classes[0]==1:
                data[i+3,6] = t.value[i][0][0]
            else:
                Sum = sum(t.value[i][0])

                if Sum>0:
                    data[i+3,6] = t.value[i][0][0]/Sum
                else:
                    data[i+3,6] = 0 

                if (data[i+3,6]>1):
                    print data[i+3,6]
    else:
        for i in range(t.capacity):
            data[i+3,3] = i
            data[i+3,4] = t.feature[i]
            data[i+3,5] = t.threshold[i]   
            data[i+3,6] = t.value[i][0][0]*clf.learning_rate*clf.n_estimators
    
    df = pd.DataFrame(data,columns=['RFID','TreeID','DataTypeID','Ind','Feature','Threshold','Probability'])
    df.to_sql('RFdata',con,index=False,if_exists='append')  
    con.commit()
    #return df

# <codecell>

def TreeTest2Txt(clf,X,fileName):
    f = open(fileName, 'w+')
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    if (type(clf) ==  type(GBR())):
        proba = clf.predict(X)
        for i in range(X.shape[0]):
            s= ''
            for j in range(X.shape[1]):
                s +='%s;' %  str(X.ix[i,j])
            s +='%s;' %  str(proba[i])
            f.write(s+'\n')
        f.close()
        return
    
    if (type(clf) ==  type(GBC())):
        proba = clf.predict_proba(X)
        for i in range(X.shape[0]):
            s= ''
            for j in range(X.shape[1]):
                s +='%s;' %  str(X.ix[i,j])
            for j in range(proba.shape[1]):
                s +='%s;' %  str(proba[i,j])
            f.write(s+'\n')
        f.close()
        return

    #regression
    if (clf.estimators_[0].tree_.n_classes[0]==1):
        proba = clf.predict(X)
        for i in range(X.shape[0]):
            s= ''
            for j in range(X.shape[1]):
                s +='%s;' %  str(X.ix[i,j])
            s +='%s;' %  str(proba[i])
            f.write(s+'\n')
        f.close()
        return
    
    #classification
    proba = clf.predict_proba(X)
    for i in range(X.shape[0]):
        s= ''
        for j in range(X.shape[1]):
            s +='%s;' %  str(X.ix[i,j])
        for j in range(proba.shape[1]):
            s +='%s;' %  str(proba[i,j])
        f.write(s+'\n')
    f.close()
    return

def TreeTest2Sql(clf,X,RFID,Type,Side,con):
    if (Type==0):
        proba = clf.predict_proba(X)
        df = pd.DataFrame(proba[:,0], columns = ['Probability'])
    else:
        proba = clf.predict(X)
        df = pd.DataFrame(proba, columns = ['Probability'])
    df["RFID"] = RFID
    df["Side"] = Side
    df["Ind"] = range(X.shape[0])
    df.to_sql('TestResults',con,index=False,if_exists='append')
    con.commit()
    
def TestData2Sql(side,X,y,db):
    import sqlite3 as lite
    import sys
        
    con = None
    
    try:
        con = lite.connect(db, timeout=10)
        cur = con.cursor()    
        cur.execute('SELECT SQLITE_VERSION()')
        con.commit()
        data = cur.fetchone()
        print "SQLite version: %s" % data  
        
    except lite.Error, e:
        print "Error %s:" % e.args[0]
        con.close()
        return     
        
    df = X.copy()
    df.columns = ['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17']
    df["Side"] = side
    df["YSide"] = 0
    df.YSide[y>0] = 1
    df.YSide[y<0] = -1
    df["Ind"] = range(df.shape[0])
    df["IndPos"] = -1
    df["IndNeg"] = -1
    df.IndPos[(df.Side==1)  & (df.YSide==-1)] = range(df.YSide[(df.Side==1)  & (df.YSide==-1)].shape[0])
    df.IndNeg[(df.Side==-1) & (df.YSide==-1)] = range(df.YSide[(df.Side==-1) & (df.YSide==-1)].shape[0])
    df.to_sql('TestFeatures',con,index=False,if_exists='append')
    con.commit()

# <codecell>

def Forest2Txt(clf,X,Dir):
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    if ((type(clf) ==  type(GBR())) | (type(clf) ==  type(GBC()))):
        for i in range(clf.n_estimators):
            Tree2Txt(clf,clf.estimators_[i][0].tree_,Dir + '/%u.t' % i)
    else:
        for i in range(clf.n_estimators):
            Tree2Txt(clf,clf.estimators_[i].tree_,Dir + '/%u.t' % i)
    TreeTest2Txt(clf,X,Dir + '/test.u')
    
def Forest2Sql(clf,X,Type,Side,Advantage,Precision,db):
    #Create connection
    import sqlite3 as lite
    import sys
    import pandas as pd
    import time
        
    con = None
    
    try:
        con = lite.connect(db, timeout=10)
        cur = con.cursor()    
        cur.execute('SELECT SQLITE_VERSION()')
        con.commit()
        data = cur.fetchone()
        print "SQLite version: %s" % data  
        
    except lite.Error, e:
        print "Error %s:" % e.args[0]
        con.close()
        return     
        
    #Clear data for this RF
    datatypes = pd.read_sql("SELECT * FROM RFdatatypes", con)
    con.commit()

    df = pd.read_sql("SELECT * FROM RFnames", con)
    con.commit()
      
    if df.shape[0]>0:
        RFID = df.ID.max()+1
    else: 
        RFID = 0
        
    df = pd.DataFrame(columns=['ID','Type','Side','NTrees','Advantage'])
    
    #regression
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    if (type(clf) ==  type(GBR())):
        Type = 1
        Advantage = 0
    else:
        if (clf.estimators_[0].tree_.n_classes[0]==1):
            Type = 1
            Advantage = 0
        #classification
        else:
            Type = 0
    
    df = df.append({'ID':RFID,'Type':Type,'Side':Side,'NTrees':clf.n_estimators,'Advantage':Advantage,'Precision':Precision},ignore_index=True) 
    df.to_sql('RFnames',con,index=False,if_exists='append')
    con.commit()
        
    #Write RF data 
    if ((type(clf) ==  type(GBR())) | (type(clf) ==  type(GBC()))):
        for i in range(clf.n_estimators):
            Tree2Sql(clf,clf.estimators_[i][0].tree_,RFID, i,datatypes,con)
    else:
        for i in range(clf.n_estimators):
            Tree2Sql(clf,clf.estimators_[i].tree_,RFID, i, datatypes,con)
    
    TreeTest2Sql(clf,X,RFID,Type,Side,con)
    
    con.commit()
    con.close()   

# <codecell>

def DropDB(db):
    import sqlite3 as lite
    import sys
        
    con = None
    
    try:
        con = lite.connect(db, timeout=10)
        cur = con.cursor()    
        cur.execute('SELECT SQLITE_VERSION()')
        con.commit()
        data = cur.fetchone()
        print "SQLite version: %s" % data  
        
    except lite.Error, e:
        print "Error %s:" % e.args[0]
        con.close()
        return     
    
    cur = con.execute("DELETE FROM RFdata")
    con.commit()
    cur = con.execute("DELETE FROM RFnames")
    con.commit()
    cur = con.execute("DELETE FROM TestFeatures")
    con.commit()
    cur = con.execute("DELETE FROM TestResults")
    con.commit()
    con.close()

# <codecell>

def visualize_tree(clf):
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    if (type(clf) !=  type(GBR())):
        t=clf.estimators_[9].tree_
    else:
        t=clf.estimators_[9][0].tree_
    from sklearn.externals.six import StringIO  
    import pydot
    from sklearn import tree
    out = StringIO() 
    tree.export_graphviz(t, out_file=out) 
    #print out.getvalue()
    graph = pydot.graph_from_dot_data(out.getvalue()) 
    graph.write_pdf("t.pdf") 

# <codecell>

def create_featuresCumulative(df,nImb):
    import pandas as pd
    import numpy 
    
    stocks = set(df.Symbol)
    dates = set(df.Date)
    
    #fdf = pd.DataFrame(columns=['Ask','AskD','Near','Far','Spread',
    #'D5', 'D555', 'D66', 'V1','V1n', 'V11', 'V11n',
    #'V8','V8n','V8nn', 'a1','a4','a5','a14','y'])
    
    fdf = numpy.zeros((len(stocks)*len(dates),20))
    
    n = 0
    for date in dates:
        print date
        for stock in stocks:
            
            #if (n>10):
            #   return fdf  
            
            #print stock
            
            stockData = df[(df.Symbol==stock) & (df.Date==date)]
    
            closeP = np.array(stockData[stockData.tSide=='YDAY'].tPrice)
            if (len(closeP)==0):
                continue;
            else:
                closeP = closeP[0]
            
            openP = np.array(stockData[stockData.Reason=='OPG'].tPrice)
            if (len(openP)==0):
                continue;
            else:
                openP = openP[0]
            
            if (closeP==0):  
                continue;
            if (openP==0):   
                continue;

            bid = np.array(stockData[stockData.Reason=='Imbalance'].Bid_P)
            if (len(bid)!=24):
                if (len(bid)!=0):
                    print stock, len(bid)
                continue;
                                
            ask = np.array(stockData[stockData.Reason=='Imbalance'].Ask_P)    
            midP = 0.5*(bid + ask)
            spread = ask - bid
            bidS = np.array(stockData[stockData.Reason=='Imbalance'].Bid_S)
            askS = np.array(stockData[stockData.Reason=='Imbalance'].Ask_S)  
            spreadS = askS - bidS
            midS = 0.5*(bidS + askS)

            ref = np.array(stockData[stockData.Reason=='Imbalance'].ImbRef)  
            near = np.array(stockData[stockData.Reason=='Imbalance'].ImbCBC)  
            far = np.array(stockData[stockData.Reason=='Imbalance'].ImbFar) 
            
            imbShares = np.array(stockData[stockData.Reason=='Imbalance'].ImbShares)  
            imbPaired = np.array(stockData[stockData.Reason=='Imbalance'].ImbPaired)  
            #====================================================================
            #Creating features
            #====================================================================
            
            if (ref[nImb]<=0):                  continue;
            if (near[nImb]<=0):                  continue;   
            if (far[nImb]<=0):                  continue;  
            if (ask[nImb]<=0):                  continue;    
            if (bid[nImb]<=0):                  continue;    
    
            
            tmpBid = bid[nImb]/ref[nImb]-1
               
            tmpAsk = ask[nImb]/ref[nImb]-1   

            #fdf['BidD'] = bid-ref

            tmpAskD = ask[nImb] - ref[nImb]   

            tmpa1 = tmpBid*tmpAsk        

            tmpNear = near[nImb]/ref[nImb]-1

            tmpFar = far[nImb]/ref[nImb]-1

            #fdf['PrevCLC'] = closeP/ref-1

            tmpSpread = spread[nImb]/ref[nImb]

            #fdf['D3'] = 100*(midP/closeP-1)

            tmpD4 = 100*(bid[nImb]-ref[nImb])/closeP

            tmpD5 = 100*(ref[nImb]-ask[nImb])/closeP

            tmpD444 = (bid[nImb]-ref[nImb])/(1+spread[nImb])

            tmpD555 = (ref[nImb]-ask[nImb])/(1+spread[nImb])

            #fdf['D7'] = 100*(ref/closeP-1)

            tmpD66 = 100*(ref[nImb]/midP[nImb]-1)

            tmpV1 = numpy.sign(imbShares[nImb])*\
            (askS[nImb] - bidS[nImb])/(100+np.abs(imbShares[nImb]))

            tmpV1n = numpy.sign(imbShares[nImb])*spreadS[nImb]/(midS[nImb]+np.abs(imbShares[nImb]))

            tmpV11 = numpy.sign(imbShares[nImb])*spreadS[nImb]/(100+imbPaired[nImb])

            tmpV11n =numpy.sign(imbShares[nImb])*spreadS[nImb]/(midS[nImb]+imbPaired[nImb])

            tmpV8 = imbShares[nImb]/(100+imbPaired[nImb])

            tmpV8n = imbShares[nImb]/(midS[nImb]+imbPaired[nImb])

            tmpV8nn = (imbShares[nImb]-spreadS[nImb])/(midS[nImb]+imbPaired[nImb])

            #fdf['a3'] = fdf['D3']*fdf['D4']

            tmpa4 = tmpD5*tmpD4

            tmpa5 = tmpD444*tmpD555

            #fdf['a6'] = fdf['D444']*fdf['D444']

            #fdf['a7'] = fdf['D555']*fdf['D555']

            tmpa14 = np.sign(imbShares[nImb])
        
            if (imbShares[nImb]>0):
                tmpy = openP>ask[nImb]
            else:
                if (imbShares[nImb]<0):
                    tmpy = openP<bid[nImb]
                else:
                    if (imbShares[nImb]==0):
                        tmpy = 0
        
            fdf[n,:] = [tmpAsk,tmpAskD,tmpNear,tmpFar,tmpSpread,\
                        tmpD5,tmpD555,tmpD66,tmpV1,tmpV1n,\
                        tmpV11,tmpV11n,tmpV8,tmpV8n,tmpV8nn,\
                        tmpa1,tmpa4,tmpa5,tmpa14,tmpy]
            
            n+=1
        
            # &
            #(imbalanceMsg.Ask_P - imbalanceMsg.Bid_P < 
            # 1.0 * (imbalanceMsg.Ask_P + imbalanceMsg.Bid_P)*0.5) &
            #(imbShares!=0)
                
            #X_pos=X_pos[['Ask','AskD','Near','Far','Spread',
            # 'D5', 'D555', 'D66', 'V1','V1n', 'V11', 'V11n',
            # 'V8','V8n','V8nn', 'a1','a4','a5']]
            
        return fdf[:n,:]   

# <codecell>

def create_featuresCumulative2(imbalanceMsg):  
    #Creating features for algorithm
    import numpy
    fdf = pd.DataFrame()

    midP = 0.5*(imbalanceMsg.Ask_P + imbalanceMsg.Bid_P)
    bid = imbalanceMsg.Bid_P
    bidS = imbalanceMsg.Bid_S   
    ref = imbalanceMsg.ImbRef
    ask = imbalanceMsg.Ask_P
    askS = imbalanceMsg.Ask_S
    near = imbalanceMsg.ImbCBC
    far = imbalanceMsg.ImbFar
    closeP = imbalanceMsg.PrevCLC_P
    
    fdf['Symbol'] = imbalanceMsg.Symbol
    fdf['Date'] = imbalanceMsg.Date

    fdf['Move'] = imbalanceMsg.Move
    
    fdf['Pnl'] = imbalanceMsg.OPC_P/imbalanceMsg.ImbRef-1
    
    fdf['MoveR'] = (imbalanceMsg.OPC_P>imbalanceMsg.ImbRef)*(imbalanceMsg.ImbShares>0)+\
     (imbalanceMsg.OPC_P<imbalanceMsg.ImbRef)*(imbalanceMsg.ImbShares<0)     
    
    fdf['CMoveR'] = imbalanceMsg.Move
    fdf.CMoveR = 0
    fdf.CMoveR[(imbalanceMsg.ImbShares>0) & (imbalanceMsg.OPC_P>imbalanceMsg.ImbRef)] = 2
    fdf.CMoveR[(imbalanceMsg.ImbShares>0) & (imbalanceMsg.OPC_P<=imbalanceMsg.ImbRef)\
              & (imbalanceMsg.OPC_P>=imbalanceMsg.Ask_P)] = 1
    fdf.CMoveR[(imbalanceMsg.ImbShares<0) & (imbalanceMsg.OPC_P<imbalanceMsg.ImbRef)] = -2
    fdf.CMoveR[(imbalanceMsg.ImbShares<0) & (imbalanceMsg.OPC_P>=imbalanceMsg.ImbRef)\
              & (imbalanceMsg.OPC_P<=imbalanceMsg.Bid_P)] = -1   
    
    fdf['CMove'] = imbalanceMsg.Move
    fdf.CMove = 0
    fdf.CMove[imbalanceMsg.OPC_P-imbalanceMsg.Ask_P>0.1] = 2
    fdf.CMove[(imbalanceMsg.OPC_P-imbalanceMsg.Ask_P>0)
              & (imbalanceMsg.OPC_P-imbalanceMsg.Ask_P<=0.1)] = 1
    fdf.CMove[(imbalanceMsg.Bid_P-imbalanceMsg.OPC_P>0.1)] = -2
    fdf.CMove[(imbalanceMsg.Bid_P-imbalanceMsg.OPC_P>0)
              & (imbalanceMsg.Bid_P-imbalanceMsg.OPC_P<=0.1)] = -1           
        
    fdf['Bid'] = bid/ref-1
    
    fdf['Ask'] = ask/ref-1   
        
    fdf['BidD'] = bid-ref
    
    fdf['AskD'] = ask - ref    
    
    fdf['a1'] = fdf['Bid']*fdf['Ask']        
       
    fdf['Near'] = near/ref-1
    
    fdf['Far'] = far/ref-1
        
    fdf['PrevCLC'] = closeP/ref-1
    
    fdf['Spread'] = (ask - bid)/ref

    fdf['D3'] = 100*(midP/closeP-1)
    
    fdf['D4'] = 100*(bid-ref)/closeP

    fdf['D5'] = 100*(ref-ask)/closeP
    
    fdf['D444'] = (bid-ref)/(1+ask - bid)

    fdf['D555'] = (ref-ask)/(1+ask - bid)
    
    fdf['D7'] = 100*(ref/closeP-1)
    
    fdf['D66'] = 100*(ref/midP-1)

    fdf['V1'] = numpy.sign(imbalanceMsg.ImbShares)*(askS - bidS)/(100+np.abs(imbalanceMsg.ImbShares))
    
    fdf['V1n'] = numpy.sign(imbalanceMsg.ImbShares)*(askS - bidS)/((askS + bidS)/2+np.abs(imbalanceMsg.ImbShares))
    
    fdf['V11'] = numpy.sign(imbalanceMsg.ImbShares)*(askS - bidS)/(100+imbalanceMsg.ImbPaired)
    
    fdf['V11n'] =numpy.sign(imbalanceMsg.ImbShares)*(askS - bidS)/((askS + bidS)/2+imbalanceMsg.ImbPaired)
    
    fdf['V8'] = imbalanceMsg.ImbShares/(100+imbalanceMsg.ImbPaired)
    
    fdf['V8n'] = imbalanceMsg.ImbShares/((askS + bidS)/2+imbalanceMsg.ImbPaired)
    
    fdf['V8nn'] = (imbalanceMsg.ImbShares-(askS - bidS))/((askS + bidS)/2+imbalanceMsg.ImbPaired)
         
    fdf['a3'] = fdf['D3']*fdf['D4']
    
    fdf['a4'] = fdf['D5']*fdf['D4']
    
    fdf['a5'] = fdf['D444']*fdf['D555']
    
    fdf['a6'] = fdf['D444']*fdf['D444']
    
    fdf['a7'] = fdf['D555']*fdf['D555']
            
    fdf['a14'] = np.sign(imbalanceMsg.ImbShares)
    
    fdf.index = range(fdf.shape[0])
    
    return fdf, Features

# <codecell>

def createFeatures34(fdf):
    import itertools
    initNumFeatures = fdf.shape[1]
    diffsInd = [x for x in itertools.combinations(range(initNumFeatures),2)]
    d_fdf = pd.DataFrame()
    for i in range(len(diffsInd)):
        #print i, diffsInd[i]
        d_fdf["d"+str(i)] = fdf.ix[:,diffsInd[i][0]]-fdf.ix[:,diffsInd[i][1]]
    return d_fdf
def createFeatures35(X_pos):
    dfdf=createFeatures34(X_pos)
    df=pd.DataFrame()
    for i in range(X_pos.shape[1]):
        for j in range(dfdf.shape[1]):
            tmp=X_pos.ix[:,i]
            tmp.ix[tmp.ix[:]==0]=1
            df[str(j)+str(i)]=dfdf.ix[:,j]/X_pos.ix[:,i]
    return pd.concat([dfdf,df], axis=1)

def createFeaturesBench(X_pos):
    X_pos_Benchmark = pd.DataFrame()
    X_pos_Benchmark['Ask'] = X_pos.Ask/X_pos.Ref-1
    X_pos_Benchmark['Bid'] = X_pos.Bid/X_pos.Ref-1
    X_pos_Benchmark['AskD'] = X_pos.Ask-X_pos.Ref
    X_pos_Benchmark['Near'] = X_pos.Near/X_pos.Ref-1
    X_pos_Benchmark['Far'] = X_pos.Far/X_pos.Ref-1
    X_pos_Benchmark['Spread'] = X_pos.Ask-X_pos.Bid
    X_pos_Benchmark['D4'] = 100*(X_pos.Bid-X_pos.Ref)/X_pos.CloseP
    X_pos_Benchmark['D5'] = 100*(X_pos.Ref-X_pos.Ask)/X_pos.CloseP

    X_pos_Benchmark['D444'] = (X_pos.Bid-X_pos.Ref)/(1+X_pos_Benchmark['Spread'])
    X_pos_Benchmark['D555'] = (X_pos.Ref-X_pos.Ask)/(1+X_pos_Benchmark['Spread'])
    X_pos_Benchmark['D66'] = 100*(X_pos.Ref/(0.5*(X_pos.Ask+X_pos.Bid))-1)
    X_pos_Benchmark['a1'] = X_pos_Benchmark['Ask']*X_pos_Benchmark['Bid']
    X_pos_Benchmark['a4'] = X_pos_Benchmark['D4']*X_pos_Benchmark['D5']
    X_pos_Benchmark['a5'] = X_pos_Benchmark['D444']*X_pos_Benchmark['D555']
    
    X_pos_Benchmark['Spy1'] = (X_pos.BidSPY+X_pos.AskSPY)/X_pos.ClosePSPY
    X_pos_Benchmark['Spy1'] /= (X_pos.Bid+X_pos.Ask)/X_pos.CloseP  
    #X_pos_Benchmark['Spy1'] /= (X_pos.Bid-X_pos.Ask)
    #X_pos_Benchmark['Spy1'] *= (X_pos.BidSPY-X_pos.AskSPY)

    return X_pos_Benchmark

def autocreateFeatures(X_pos,y_pos,ind):
    from sklearn.ensemble import RandomForestClassifier as RF
    clf = RF(min_samples_split = X_pos.shape[0]*0.05, criterion = 'entropy',n_estimators =10)
    if ind==1:
        newX_pos=createFeatures35(X_pos)
    else:
        newX_pos=createFeatures34(X_pos)
    
    clf.fit(newX_pos,y_pos)

    fi = pd.DataFrame()
    fi['Feature'] = list(newX_pos.columns)
    fi['Impotrance'] = clf.feature_importances_
    fi=fi.sort(columns=['Impotrance'],ascending=False)
    fi['Index'] = range(newX_pos.shape[1])
    fi.index = fi['Index']

    for i in range(fi.shape[0]):
        if (fi['Impotrance'][i]<0.005):
            break
        #print fi['Feature'][i]

    newX_pos = newX_pos[fi['Feature'][:i]]

    #Stage 2
    poly = PolynomialFeatures(2)
    newX_pos_2=pd.DataFrame(poly.fit_transform(newX_pos))
    clf.fit(newX_pos_2,y_pos)

    fi = pd.DataFrame()
    fi['Feature'] = list(newX_pos_2.columns)
    fi['Impotrance'] = clf.feature_importances_
    fi=fi.sort(columns=['Impotrance'],ascending=False)
    fi['Index'] = range(newX_pos_2.shape[1])
    fi.index = fi['Index']

    for i in range(fi.shape[0]):
        if (fi['Impotrance'][i]<0.01):
            break
        #print fi['Feature'][i]

    newX_pos_2 = newX_pos_2[fi['Feature'][:i]]

    return newX_pos_2

