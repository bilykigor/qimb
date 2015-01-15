# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

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
    import pandas as pd
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
    import numpy as np
    import pandas as pd
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
    
    graph = pydot.graph_from_dot_data(out.getvalue()) 
    graph.write_pdf("t.pdf") 

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

def Forest2SqlReg(db,fdf,X_pos,X_neg):
    #Random forest for regression
    
    from sklearn.ensemble import RandomForestRegressor as RFR
    '''Xpp = X_pos.copy()
    Xpp = Xpp[ypln_pos>0]
    Xpp.index = range(Xpp.shape[0])

    clf = RFR(min_samples_split = ypln_pos[ypln_pos>0].shape[0]*0.05)
    clf.fit(Xpp, ypln_pos[ypln_pos>0])
    qimbs.Forest2Txt(clf, Xpp.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/PP')

    clf = GBR(min_samples_split = ypln_pos[ypln_pos>0].shape[0]*0.05, loss='huber',init='zero',learning_rate=0.1)
    clf.fit(Xpp, ypln_pos[ypln_pos>0])
    qimbs.Forest2Txt(clf, Xpp.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/PP')'''

    #------------------------------------------
    Xpn = X_pos.copy()
    Xpn = Xpn[ypln_pos<0]
    Xpn.index = range(Xpn.shape[0])
    
    clf, r2 = run_reg(X_pos,ypln_pos,RFR,20,20,dates,datesDF_pos)
    qimbs.Forest2Txt(clf, Xpn.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/PN')
    qimbs.Forest2Sql(clf, Xpn.ix[:,:],1,1,0,r2,db)

    clf, r2 = run_reg(X_pos,ypln_pos,GBR,20,20,dates,datesDF_pos)
    qimbs.Forest2Txt(clf, Xpn.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/PN')
    qimbs.Forest2Sql(clf, Xpn.ix[:,:],1,1,0,r2,db)

    #------------------------------------------
    '''Xnp = X_neg.copy()
    Xnp = Xnp[ypln_neg>0]
    Xnp.index = range(Xnp.shape[0])

    clf = RFR(min_samples_split = ypln_neg[ypln_neg>0].shape[0]*0.05)
    clf.fit(Xnp, ypln_neg[ypln_neg>0])
    qimbs.Forest2Txt(clf, Xnp.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/NP')

    clf = GBR(min_samples_split = ypln_neg[ypln_neg>0].shape[0]*0.05, loss='huber',init='zero',learning_rate=0.1)
    clf.fit(Xnp, ypln_neg[ypln_neg>0])
    qimbs.Forest2Txt(clf, Xnp.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/NP')'''

    #------------------------------------------
    Xnn = X_neg.copy()
    Xnn = Xnn[ypln_neg<0]
    Xnn.index = range(Xnn.shape[0])

    clf, r2 = run_reg(X_neg,ypln_neg,RFR,20,20,dates,datesDF_neg)
    qimbs.Forest2Txt(clf, Xnn.ix[:,:],'/home/user1/Desktop/Share2Windows/Forest/NN')
    qimbs.Forest2Sql(clf, Xnn.ix[:,:],1,-1,0,r2,db)

    clf, r2 = run_reg(X_neg,ypln_neg,GBR,20,20,dates,datesDF_neg)
    qimbs.Forest2Txt(clf, Xnn.ix[:,:],'/home/user1/Desktop/Share2Windows/GradientBoost/NN')
    qimbs.Forest2Sql(clf, Xnn.ix[:,:],1,-1,0,r2,db)

