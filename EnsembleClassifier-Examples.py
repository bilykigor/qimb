# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from ggplot import *
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline 
from sklearn.lda import LDA

# <codecell>

#Examples
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

# <codecell>

#Single estimators
np.random.seed(123)
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()
numFolds = 20

print(str(numFolds)+'-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3], ['Logistic Regression', 'Random Forest', 'naive Bayes']):
    scores = cross_validation.cross_val_score(clf, X, y, cv=numFolds, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# <codecell>

#Single estimators vs EnsembleClassifier
from EnsembleClassifier import EnsembleClassifier
np.random.seed(123)
eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3])

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_validation.cross_val_score(clf, X, y, cv=numFolds, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# <codecell>

#EnsembleClassifier with weights
np.random.seed(123)

df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))

i = 0
for w1 in range(1,4):
    for w2 in range(1,4):
        for w3 in range(1,4):
            
            if len(set((w1,w2,w3))) == 1: # skip if all weights are equal
                continue
            
            eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], weights=[w1,w2,w3])
            scores = cross_validation.cross_val_score(
                                            estimator=eclf,
                                            X=X, 
                                            y=y, 
                                            cv=5, 
                                            scoring='accuracy',
                                            n_jobs=1)
            
            df.loc[i] = [w1, w2, w3, scores.mean(), scores.std()]
            i += 1
            
df=df.sort(columns=['mean', 'std'], ascending=False)
df['label'] = df.index

ggplot(df, aes(x='mean',y='std', label='label')) + geom_point()# + geom_text(df, aes(x='mean',y='std', label='label'))

# <codecell>

#EnsembleClassifier in Pipeline
from EnsembleClassifier import ColumnSelector
pipe1 = Pipeline([
               ('sel', ColumnSelector([1])),    # use only the 1st feature
               ('clf', GaussianNB())])

pipe2 = Pipeline([
               ('sel', ColumnSelector([0, 1])), # use the 1st and 2nd feature
               ('dim', LDA(n_components=1)),    # Dimensionality reduction via LDA
               ('clf', LogisticRegression())])

eclf = EnsembleClassifier([pipe1, pipe2])
scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# <codecell>


