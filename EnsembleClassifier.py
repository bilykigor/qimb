# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

# <codecell>

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """ 
    Ensemble classifier for scikit-learn estimators.
        
    Parameters
    ----------
    
    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list`
      Weights for the individual classifiers for `.predict_proba`. 
      Using equal weight by default.
      
    """
    def __init__(self, clfs=None, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        """ 
        Fit the scikit-learn estimators.
        
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels
      
        """
        for clf in self.clfs:
            clf.fit(X, y)
            
    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
        
        Returns
        ----------
        
        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule      
        """
        
        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)
        
        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])
        
        return maj
            
    def predict_proba(self, X):
        
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
        
        Returns
        ----------
        
        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.
        
        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)
        
        return avg
    
class ColumnSelector(object):
    """ 
    A feature selector for scikit-learn's Pipeline class that returns
    specified columns from a numpy array.
    
    """
    
    def __init__(self, cols):
        self.cols = cols
        
    def transform(self, X, y=None):
        return X[:, self.cols]

    def fit(self, X, y=None):
        return self    

# <codecell>

class LabeledEstimator(BaseEstimator, ClassifierMixin):
    """ 
    For each unique label trains estimator of type clf_parent and use its prediction
    """
    
    import numpy as np
    
    def __init__(self, clf_parent=None):
        self.clf_parent = clf_parent
        self.clfs = dict()
        
    def fit(self, X, y, labels = None):   
        from sklearn.base import clone
        
        if type(self.clf_parent)==type(None):
            return
                
        if type(labels)==type(None):
            self.clfs['None'] = clone(self.clf_parent)
            self.clfs['None'].fit(X, y)
        else:
            for label in set(labels):
                self.clfs[label] = clone(self.clf_parent)
                self.clfs[label].fit(X.loc[labels==label], y.loc[labels==label])
            
    def predict(self, X, labels = None): 
        md = dict()
        md['None'] = self.predictMaj(X)
        for key in self.clfs.keys():
            md[key] = self.clfs[key].predict(X)
            
        if type(labels)==type(None):
            return np.asarray(md['None'])
        else:           
            f = lambda md,key,i: md[key][i] if md.has_key(key) else md['None'][i]
            
            return np.asarray([f(md,key,i) for i,key in enumerate(labels)])
            
    def predictMaj(self, X, labels = None):  
        """
            Predicted class labels by majority rule      
        """
        if type(self.clf_parent)==type(None):
            return
        
        if type(labels)==type(None):
            if (len(self.clfs)>1):
                #clf for each label makes prediction 
                tmp = np.asarray([clf.predict(X) for clf in self.clfs.itervalues()])
                
                #then as a result we use class wich appeared most frequently
                return np.asarray([max(list(tmp[:,c]), key=list(tmp[:,c]).count) for c in range(tmp.shape[1])])
            else:
                for clf in self.clfs.itervalues():
                    return clf.predict(X)
        else:
            result=[]
            for index,row in X.iterrows():
                if labels[index] in self.clfs:
                    result.append(self.clfs[labels[index]].predict(row))
                else:
                    l =  [clf.predict(row) for clf in self.clfs.itervalues()]
                    
                    result.append(max(list(l), key=list(l).count)[0])

            return np.asarray(result)
            
    def predict_proba(self, X, labels = None):

        if type(self.clf_parent)==type(None):
            return

        if len(self.clfs)==0:
            return

        if type(labels)==type(None):#if no lables just get avg prediction of all clfs
            if (len(self.clfs)>1):
                allpred = [clf.predict_proba(X) for clf in self.clfs.itervalues()]
                return np.average(allpred, axis=0)
            else:
                for clf in self.clfs.itervalues():
                    return clf.predict_proba(X)
        else:
            md = dict()
            allpred = [clf.predict_proba(X) for clf in self.clfs.itervalues()]
            md['None'] = np.average(allpred, axis=0)
            for key in self.clfs.keys():
                md[key] = self.clfs[key].predict_proba(X)

            f = lambda md,key,i: md[key][i] if md.has_key(key) else md['None'][i]
            
            return np.asarray([f(md,key,i) for i,key in enumerate(labels)])

# <codecell>


