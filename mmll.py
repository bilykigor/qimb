# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

# <codecell>

def LeavePRandOut(n, p, nFolds):
    """
    Example: result = LeavePRandOut(5,2,10)
    """
    result = []
    allIndexes = range(n)
    for i in range(nFolds): 
        np.random.shuffle(allIndexes)
        test_int = np.asarray(allIndexes[:p])
        train_ind = np.asarray(allIndexes[p:])
        result.append((train_ind,test_int))
        
    return result

def LeavePRandLabelOut1(labels, p, nFolds):
    """
    Example: 
    
    labels = [0,0,1,1,1,2,3,4,5]
    result = LeavePRandLabelOut(labels,2,10)
    """
    result = []
    
    uniqueLabels = list(set(labels))
    allIndexes = range(len(labels))
    labelsIndexes = range(len(uniqueLabels))
    
    for k in range(nFolds): 
        np.random.shuffle(labelsIndexes)
        test_labels = [uniqueLabels[i] for i in labelsIndexes[:p]]
        test_int = np.asarray([i for i,v in enumerate(labels) if v in test_labels])
        train_ind = np.asarray(list(set(allIndexes)-set(test_int)))
        result.append((train_ind,test_int))
        
    return result

def LeavePRandLabelOut2(labels, p, nFolds):
    """
    Example: 
    
    labels = [0,0,1,1,1,2,3,4,5]
    result = LeavePRandLabelOut(labels,2,10)
    """
    from itertools import combinations
    
    result = []
    
    testLabelsCombs = list(combinations(set(labels), p))#this is to heavy opearation when set(labels) is large
    np.random.shuffle(testLabelsCombs)

    testLabelsCombs = testLabelsCombs[:min(nFolds,len(testLabelsCombs))]
    
    return zip([np.asarray([i for i,v in enumerate(labels) if v not in testLabels]) for testLabels in testLabelsCombs],
    [np.asarray([i for i,v in enumerate(labels) if v in testLabels]) for testLabels in testLabelsCombs])

def LeavePRandLabelOut(labels, p, nFolds):
    from math import factorial
   
    n = len(set(labels))
    
    chosenk = factorial(n)/(factorial(n-p)*factorial(p))

    if chosenk<500000:
        return LeavePRandLabelOut2(labels, p, nFolds)
    else:
        return LeavePRandLabelOut1(labels, p, nFolds)

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

def draw_confusion_matrix(conf_arr, labels, fig, ax):  
    conf_arr=conf_arr.astype(float)
    
    sums = conf_arr.sum(axis=0)

    for i in range(len(labels)):
        conf_arr[:,i] /= sums[i]

    res = ax.imshow(np.array(conf_arr), cmap=plt.cm.jet, interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y])[:4], xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])

# <codecell>

def clf_cross_validation(clf,X,y,test_size=None,n_folds=1,labels=None,verbose=True):
    """
        when labels!=None
            dont split data with same label in train/test sets 
    """
    
    from sklearn.metrics import confusion_matrix
    from mmll import draw_confusion_matrix
    
    n = X.shape[0]
    
    if type(test_size)==type(None):
        test_size = int(0.1*n)
           
    if type(labels)!=type(None):
        lpl = LeavePRandLabelOut(labels,test_size, n_folds)
    else:
        lpl = LeavePRandOut(n, test_size, n_folds)

    y_labels =  np.sort(list(set(y)))
    n_y_labels = len(y_labels)
    
    test_cm = np.zeros((n_y_labels,n_y_labels))
    train_cm = np.zeros((n_y_labels,n_y_labels))
    
    if type(clf)!=str: 
        CLF_BEST = clf
    TEST_F_Score = 0

    for train_index, test_index in lpl:
        Xtrain, Xtest = X.loc[train_index], X.loc[test_index]
        ytrain, ytest = y.loc[train_index], y.loc[test_index]
        #print len(set.intersection(set(labels[test_index]),set(labels[train_index])))

        clf.fit(Xtrain,ytrain)

        ypred = clf.predict(Xtest)

        test_cm_tmp = confusion_matrix(ytest,ypred,y_labels).astype(float)
        test_cm += test_cm_tmp/n_folds

        #save best performance
        pr =  Precision_Recall(test_cm_tmp, y_labels)
        if (pr[2]>TEST_F_Score):
            TEST_F_Score =  pr[2]
            CLF_BEST = clf   
    
    test_pr = Precision_Recall(test_cm, y_labels)
    
    print 'Precision - %.2f, Recall - %.2f, F_Score - %.2f, max F_Score - %.2f' % (test_pr[0],test_pr[1],test_pr[2],TEST_F_Score)
    
    if verbose:
        fig1 = plt.figure(figsize=(15, 5))
        ax1 = fig1.add_subplot(1,3,1)
        draw_confusion_matrix(test_cm, y_labels , fig1, ax1)

    return test_cm,CLF_BEST

# <codecell>

def lclf_cross_validation(clf,X,y,_lables,test_size=None,n_folds=1,labels=None,verbose=True):
    """
        when labels!=None
            dont split data with same label in train/test sets 
    """
    
    from sklearn.metrics import confusion_matrix
    from mmll import draw_confusion_matrix
    
    n = X.shape[0]
    
    if type(test_size)==type(None):
        test_size = int(0.1*n)
           
    if type(labels)!=type(None):
        lpl = LeavePRandLabelOut(labels,test_size, n_folds)
    else:
        lpl = LeavePRandOut(n, test_size, n_folds)

    y_labels =  np.sort(list(set(y)))
    n_y_labels = len(y_labels)
    
    test_cm = np.zeros((n_y_labels,n_y_labels))
    
    if type(clf)!=str: 
        CLF_BEST = clf
    TEST_F_Score = 0

    for train_index, test_index in lpl:
        Xtrain, Xtest = X.loc[train_index], X.loc[test_index]
        ytrain, ytest = y.loc[train_index], y.loc[test_index]
        lables_train, lables_test = _lables.loc[train_index], _lables.loc[test_index]
        
        #print len(set.intersection(set(labels[test_index]),set(labels[train_index])))

        clf.fit(Xtrain,ytrain,labels=lables_train)

        ypred = clf.predict(Xtest,labels=lables_test)

        test_cm_tmp = confusion_matrix(ytest,ypred,y_labels).astype(float)
        test_cm += test_cm_tmp/n_folds

        #save best performance
        pr =  Precision_Recall(test_cm_tmp, y_labels)
        if (pr[2]>TEST_F_Score):
            TEST_F_Score =  pr[2]
            CLF_BEST = clf   
    
    test_pr = Precision_Recall(test_cm, y_labels)
    
    print 'Precision - %.2f, Recall - %.2f, F_Score - %.2f, max F_Score - %.2f' % (test_pr[0],test_pr[1],test_pr[2],TEST_F_Score)
    
    if verbose:
        fig1 = plt.figure(figsize=(15, 5))
        ax1 = fig1.add_subplot(1,3,1)
        draw_confusion_matrix(test_cm, y_labels , fig1, ax1)

    return test_cm,CLF_BEST

# <codecell>


