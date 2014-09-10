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
        
        f = '/home/user1/PyProjects/data/2014-%s-%s/CommonAggr_2014-%s-%s.csv' % (m,d,m,d)
        
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

    fdf['Symbol'] = imbalanceMsg.Symbol
    fdf['Date'] = imbalanceMsg.Date

    fdf['Move'] = imbalanceMsg.Move
    Features['Move'] = 'Move'
       
    fdf['Bid'] = imbalanceMsg.Bid_P/imbalanceMsg.ImbRef-1
    Features['Bid'] = 'Bid(9.28)'
    
    fdf['Ask'] = imbalanceMsg.Ask_P/imbalanceMsg.ImbRef-1
    Features['Ask'] = 'Ask(9.28)'
       
    fdf['Near'] = imbalanceMsg.ImbCBC/imbalanceMsg.ImbRef-1
    Features['Near'] = 'Near(9.28)'
    
    fdf['Far'] = imbalanceMsg.ImbFar/imbalanceMsg.ImbRef-1
    Features['Far'] = 'Far(9.28)'
        
    fdf['PrevCLC'] = imbalanceMsg.PrevCLC_P/imbalanceMsg.ImbRef-1
    Features['PrevCLC'] = 'PrevCLC'
    
    fdf.index = range(fdf.shape[0])
    
    return fdf, Features

# <codecell>

def get_imbelanceMSG2(df,nImb):
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

    
    imbalanceMsg = df[df.Reason == 'Imbalance'].between_time(startTime,endTime)
    imbalanceMsg = imbalanceMsg[
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
    
    prev_CLC = df[df.tSide == 'YDAY']
    prev_CLC = prev_CLC[['Symbol','tPrice','tShares']]
    prev_CLC.columns = ['Symbol','PrevCLC_P','PrevCLC_S']
    prev_CLC['Date'] = prev_CLC.index.date
    
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
    
    return imbalanceMsg   
    
    return imbalanceMsg   

# <codecell>

def Precision_Recall(cm):
    m = cm.shape[0]
    sums1 = cm.sum(axis=1);
    sums2 = cm.sum(axis=0);
    precision = 0
    s1 = 0
    s2 = 0
    for i in range(1,m):
        precision +=  cm[i,i]
        s1 += sums1[i]
        s2 += sums2[i]

    return precision/s2, precision/s1, 2*(precision/s1 * precision/s2)/(precision/s2 + precision/s1)

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
    
    labels = list(set(y))
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

                #scaler = preprocessing.StandardScaler().fit(Xtrain_sub)
                if (type(clf_class()) ==  type(LR())) | (type(clf_class()) ==  type(SVC())):
                    clf = clf_class(class_weight='auto')
                else:
                    clf = clf_class(min_samples_split = 20)
                    
                clf.fit(Xtrain_sub,ytrain_sub)
                #print clf.coef_
                #print '-------------------'


                ypred += clf.predict(Xtest).astype(float)/n_ensembles
                ypredTrain += clf.predict(Xtrain).astype(float)/n_ensembles

            #Averaging of assemblies results
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
        

        test_cm += confusion_matrix(ytest,ypred,labels).astype(float)/n_folds
        train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds
        #testError += np.mean(ypred != ytest)/n_folds
        #trainError += np.mean(ypredTrain != ytrain)/n_folds
        
    test_pr = Precision_Recall(test_cm)
    train_pr = Precision_Recall(train_cm)    
    return 1-train_pr[2], 1-test_pr[2], test_cm

def OneModelResults(clf_class, input,target,ERRORS,dates,datesDF,**kwargs):
    fig1 = plt.figure(figsize=(15, 5))
    plt.clf()
    ax1 = fig1.add_subplot(1,3,1)
    trainError, testError, cm = run_cv2(input,target,clf_class,10,1,dates,datesDF,**kwargs)
    draw_confusion_matrix(cm, [0,1,-1], fig1, ax1)
    ERRORS.loc[ERRORS.shape[0]] =[str(clf_class).split('.')[-1].strip('>'),trainError,testError]
    pr = Precision_Recall(cm)
    print 'Precision - %s, Recall - %s, F_Score - %s' % (pr[0],pr[1],pr[2])

    
    #Show learning curves
    TrainError=[]
    TestError=[]
    nDays = len(dates)
    testRange = range(nDays-1)
    for i in testRange: 
        trainError, testError, cm = run_cv2(input,target,clf_class,10,i+1,dates,datesDF,**kwargs)
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
    
    return g

# <codecell>

def run_cvNN(X,y,n_folds,test_size,**kwargs):
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn import preprocessing
    import neurolab as nl
    
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    
    labels = list(set(y))
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
    
    labels = list(set(y))
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

def ComplexCLF(X,y,clf_class1,clf_class2,n_folds,test_size,dates,datesDF):
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import confusion_matrix
    
    labels = list(set(y))
    test_cm = np.zeros((len(labels),len(labels)))
    train_cm = np.zeros((len(labels),len(labels)))
    testError = 0
    trainError = 0
    
    
    for i in range(n_folds):  
        #Split train - test
        r = range(len(dates))
        np.random.shuffle(r)
        test_days = r[:test_size] 
        train_days = r[test_size:] 
               
        Xtrain = X.ix[datesDF.ix[train_days],:]
        Xtest = X.ix[datesDF.ix[test_days],:]
        ytrain = y.ix[datesDF.ix[train_days]]
        ytest = y.ix[datesDF.ix[test_days]]
        #**************************************
        
        #Train first classifier
        newX=pd.DataFrame()
        clf1_pool =[]
        for l in labels:
            clf1 = clf_class1()
            clf1.fit(Xtrain,ytrain==l)
            newX[l] = clf1.predict(Xtrain)        
            clf1_pool.append(clf1)
        #**************************************
        
        #Train second classifier
        clf2 = clf_class2()
        clf2.fit(newX,ytrain)
        #**************************************
        
        ypredTrain = clf2.predict(newX)
        train_cm += confusion_matrix(ytrain,ypredTrain,labels).astype(float)/n_folds
           
        #Get prediction  
        newX=pd.DataFrame()
        for i in range(len(labels)):
            newX[labels[i]] = clf1_pool[i].predict(Xtest)
        ypred = clf2.predict(newX)
        #**************************************
        
        test_cm += confusion_matrix(ytest,ypred,labels).astype(float)/n_folds        
            
    test_pr = Precision_Recall(test_cm)
    train_pr = Precision_Recall(train_cm)    
    return 1-train_pr[2], 1-test_pr[2], test_cm

def TwoModelsResults(clf_class1,clf_class2, input,target,ERRORS,dates,datesDF):
    fig1 = plt.figure(figsize=(15, 5))
    plt.clf()
    ax1 = fig1.add_subplot(1,3,1)
    trainError, testError, cm = ComplexCLF(input,target,clf_class1,clf_class2,20,1,dates,datesDF)
    draw_confusion_matrix(cm, [0,1,-1], fig1, ax1)
    ERRORS.loc[ERRORS.shape[0]] =\
    [str(clf_class1).split('.')[-1].strip('>') + ' + ' + str(clf_class2).split('.')[-1].strip('>'),\
     trainError,testError]
    pr = Precision_Recall(cm)
    print 'Precision - %s, Recall - %s, F_Score - %s' % (pr[0],pr[1],pr[2])

    
    #Show learning curves
    TrainError=[]
    TestError=[]
    nDays = len(dates)
    testRange = range(nDays-1)
    for i in testRange: 
        trainError, testError, cm = ComplexCLF(input,target,clf_class1,clf_class2,20,i,dates,datesDF)
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

# <codecell>

def get_signals2(imbalanceMsg,X,y,clf_class1,clf_class2,dates,datesDF):  
    import numpy
    labels = list(set(y))
    
    Signals = imbalanceMsg[['Timestamp','Symbol','Ask_P','Bid_P']]
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
        
        clf1_pool =[]
        newX=pd.DataFrame()
        for l in labels:
            clf1 = clf_class1()
            clf1.fit(Xtrain,ytrain==l)
            newX[l] = clf1.predict(Xtrain)        
            clf1_pool.append(clf1)
        
        clf2 = clf_class2()
        clf2.fit(newX,ytrain)
               
        newX=pd.DataFrame()
        for i in range(len(labels)):
            newX[labels[i]] = clf1_pool[i].predict(Xtest)

        Signals.Side[datesDF.ix[test_days]] = clf2.predict(newX)
        
    Signals.Price[Signals['Side']==1] = Signals.Ask_P[Signals['Side']==1]
    Signals.Price[Signals['Side']==-1] = Signals.Bid_P[Signals['Side']==-1]
    Signals = Signals[Signals.Side!=0]
    Signals = Signals[['Timestamp','Symbol','Price','Side']] 
    Signals.index = Signals.Timestamp
    return Signals

def get_signals1(imbalanceMsg,X,y,clf_class,dates,datesDF,**kwargs):  
    import numpy
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression as LR
    labels = list(set(y))
    
    Signals = imbalanceMsg[['Timestamp','Symbol','Ask_P','Bid_P']]
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
        
        if clf_class!='B':            
            if not kwargs.has_key('n_ensembles'):
                
                if (type(clf_class()) ==  type(LR())) | (type(clf_class()) ==  type(SVC())):
                    clf = clf_class(class_weight='auto')
                else:
                    clf = clf_class()
                
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
                        clf = clf_class()
                        
                    clf.fit(Xtrain_sub,ytrain_sub)
                    
                    ypred += clf.predict(Xtest).astype(float)/n_ensembles


                #Averaging of assemblies results
                ypred[ypred>0.5] = 1
                ypred[ypred<-0.5] = -1
                ypred[(ypred!=1) & (ypred!=-1)] = 0

                Signals.Side[datesDF.ix[test_days]] = ypred
        else:
            ypred = ytest.copy(); ypred[:] = 0
            #ypred[(Xtest.a14>0)] = 1
            #ypred[(Xtest.a14<0)] = -1
            ypred[(Xtest.a14>0) & (Xtest.D5>=0)] = 1
            ypred[(Xtest.a14<0) & (Xtest.D4>=0)] = -1
            Signals.Side[datesDF.ix[test_days]] = ypred
        
        
        
    Signals.Price[Signals['Side']==1] = Signals.Ask_P[Signals['Side']==1]
    Signals.Price[Signals['Side']==-1] = Signals.Bid_P[Signals['Side']==-1]
    Signals = Signals[Signals.Side!=0]
    Signals = Signals[['Timestamp','Symbol','Price','Side']] 
    Signals.index = Signals.Timestamp
    return Signals
        

# <codecell>

def get_performance(Signals,df,days,SymbolsInd,T,TN,add=0):
    days = sorted(list(set(Signals.index.date)))
    
    PNL=[]
    SDict=dict()
    for day in days:
        dayPnl=[]
        #print '----------------------------------'
        #print day
        #print '----------------------------------'
        curr_day_signals = Signals[day.strftime("%Y-%m-%d")]
        data = df[day.strftime("%Y-%m-%d")]
        buys = curr_day_signals[curr_day_signals.Side==1]
        buys = buys.sort(['Symbol'])
        sells = curr_day_signals[curr_day_signals.Side==-1]
        sells = sells.sort(['Symbol'])
        
        endTimestamp = pd.Timestamp('09:30:01')
        
        #print 'BUY'
        for count, row in buys.iterrows():
            startTimestamp = row[0]
            
            symbol = row[1]
            price = row[2]+add

            curr_symbol_data=data[data.Symbol==symbol].between_time(startTimestamp,endTimestamp)
            
            #AskPrices = dict()
            #for i in range(curr_symbol_data.shape[0]):
            #    if curr_symbol_data.Ask_P[i]>price: continue
            #    if (not AskPrices.has_key(curr_symbol_data.Ask_P[i])):
            #        AskPrices[curr_symbol_data.Ask_P[i]] = curr_symbol_data.Ask_S[i]
                    
            volumeTraded = 1#min(abs(curr_symbol_data.ImbShares[0]),curr_symbol_data.Ask_S[0])
                           
            OPC = curr_symbol_data.tPrice[(curr_symbol_data.Reason=='OPG')]
            #print OPC
            #continue
            pnl = volumeTraded*(OPC[0]-price)
            
            if SDict.has_key(symbol):
                SDict[symbol]+=pnl
            else:
                SDict[symbol]=pnl
            
            dayPnl.append(pnl)
            #if pnl<0:
            #    print ' BUY   %s %s shares at %s SELL at %s PNL %s' % ( symbol,volumeTraded,price,OPC[0],pnl)
            
            for s in buys.Symbol:
                T[SymbolsInd[symbol],SymbolsInd[s]]+=1.0
            for s in sells.Symbol:
                T[SymbolsInd[symbol],SymbolsInd[s]]-=1.0
            
        #print 'SELL'    
        for count, row in sells.iterrows():
            startTimestamp = row[0]

            symbol = row[1]
            price = row[2]-add

            curr_symbol_data=data[data.Symbol==symbol].between_time(startTimestamp,endTimestamp)
            
            #BidPrices = dict()
            #for i in range(curr_symbol_data.shape[0]):
            #    if curr_symbol_data.Bid_P[i]<price: continue
            #    if (not BidPrices.has_key(curr_symbol_data.Bid_P[i])):
            #        BidPrices[curr_symbol_data.Bid_P[i]] = curr_symbol_data.Bid_S[i]
            
            volumeTraded = 1#min(abs(curr_symbol_data.ImbShares[0]),curr_symbol_data.Bid_S[0])
            
            OPC = curr_symbol_data.tPrice[curr_symbol_data.Reason=='OPG']
            #print OPC
            #continue
            pnl = volumeTraded*(price-OPC[0])
            
            if SDict.has_key(symbol):
                SDict[symbol]+=pnl
            else:
                SDict[symbol]=pnl
                
            dayPnl.append(pnl)
            #if pnl<0:
            #    print ' SELL  %s %s shares at %s BUY as %s PNL %s' % ( symbol,volumeTraded,price,OPC[0],pnl) 
            
            for s in sells.Symbol:
                T[SymbolsInd[symbol],SymbolsInd[s]]+=1.0
            for s in buys.Symbol:
                T[SymbolsInd[symbol],SymbolsInd[s]]-=1.0
                
        PNL.append(np.sum(dayPnl))
        print '%s %s' % (day,np.sum(dayPnl))
        
    for i in range(T.shape[0]):
        for j in range(T.shape[0]):
            TN[i,j]=2.0*T[i,j]/(T[i,i]+T[j,j])
      
    result = pd.DataFrame()
    result['Date'] = days
    result['Pnl'] = PNL
    print '----------------------------------'
    keys = sorted(SDict, key=SDict.__getitem__)
    for key in keys[-10:]:
        print '%s %s' % (key, SDict[key])
    print '----------------------------------'
    print np.sum(PNL)
    return result

# <codecell>

def Tree2Txt(t,fileName):
    f = open(fileName, 'w+')
    f.write(str(t.n_classes[0])+'\n');
    f.write(str(t.n_features)+'\n');
    f.write(str(t.capacity)+'\n');
    for i in range(t.capacity):
        s= '%d;%f;' % (t.feature[i],t.threshold[i])
        for j in range(t.n_classes[0]):
            Sum = sum(t.value[i][0])
            if Sum>0:
                s +='%s;' %  str(t.value[i][0][j]/Sum)
            else:
                 s +='%s;' % '0'
        f.write(s+'\n')
    f.close()

def TreeTest2Txt(clf,X,fileName):
    f = open(fileName, 'w+')
    proba = clf.predict_proba(X)
    for i in range(X.shape[0]):
        s= ''
        for j in range(X.shape[1]):
            s +='%s;' %  str(X.ix[i,j])
        for j in range(proba.shape[1]):
            s +='%s;' %  str(proba[i,j])
        f.write(s+'\n')
    f.close()
    
def Forest2Txt(clf,X,Dir):
    for i in range(clf.n_estimators):
        Tree2Txt(clf.estimators_[i].tree_,Dir + '/%u.t' % i)
    TreeTest2Txt(clf,X,Dir + '/test.u')

# <codecell>

def visualize_tree(clf):
    t=clf.estimators_[0].tree_
    from sklearn.externals.six import StringIO  
    import pydot
    from sklearn import tree
    out = StringIO() 
    tree.export_graphviz(t, out_file=out) 
    #print out.getvalue()
    graph = pydot.graph_from_dot_data(out.getvalue()) 
    graph.write_pdf("t.pdf") 

