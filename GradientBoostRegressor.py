# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np

def ground_truth(x):
    """Ground truth -- function to approximate"""
    return x * np.sin(x) + np.sin(2 * x)

def gen_data(n_samples=200):
    """generate training and testing data"""
    np.random.seed(13)
    x = np.random.uniform(0, 10, size=n_samples)
    x.sort()
    y = ground_truth(x) + 0.75 * np.random.normal(size=n_samples)
    train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    x_train, y_train = x[train_mask, np.newaxis], y[train_mask]
    x_test, y_test = x[~train_mask, np.newaxis], y[~train_mask]
    return x_train, x_test, y_train, y_test
    

X_train, X_test, y_train, y_test = gen_data(200)

# plot ground truth
x_plot = np.linspace(0, 10, 500)

def plot_data(figsize=(8, 5)):
    fig = plt.figure(figsize=figsize)
    gt = plt.plot(x_plot, ground_truth(x_plot), alpha=0.4, label='ground truth')

    # plot training and testing data
    plt.scatter(X_train, y_train, s=10, alpha=0.4)
    plt.scatter(X_test, y_test, s=10, alpha=0.4, color='red')
    plt.xlim((0, 10))
    plt.ylabel('y')
    plt.xlabel('x')
    
plot_data(figsize=(8, 5))

# <codecell>

from itertools import islice
from sklearn.ensemble import GradientBoostingRegressor

plot_data()

est = GradientBoostingRegressor(n_estimators=2, max_depth=1, learning_rate=1.0,init='zero')
est.fit(X_train, y_train)

pred = est.predict(x_plot[:, np.newaxis])
plt.plot(x_plot, pred, color='b', linewidth=2)

# <codecell>

from sklearn.tree import DecisionTreeRegressor
plot_data()
for dt in est.estimators_:
    plt.plot(x_plot, dt[0].predict(x_plot[:, np.newaxis]))

# <codecell>

y=est.estimators_[0][0].predict(x_plot[:, np.newaxis])+\
  est.estimators_[1][0].predict(x_plot[:, np.newaxis])
plot_data()
plt.plot(x_plot, pred, color='b', linewidth=2)
plt.plot(x_plot, y,color='y')

# <codecell>

y[0]-pred[0]

# <codecell>

def deviance_plot(est, X_test, y_test, ax=None, label='', train_color='#2c7bb6', 
                  test_color='#d7191c', alpha=1.0):
    n_estimators = len(est.estimators_)
    
    test_dev = np.empty(n_estimators)

    for i, pred in enumerate(est.staged_predict(X_test)):
       test_dev[i] = est.loss_(y_test, pred)

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        
    ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label, 
             linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color, 
             label='Train %s' % label, linewidth=2, alpha=alpha)
    ax.set_ylabel('Error')
    ax.set_xlabel('n_estimators')
    ax.set_ylim((0, 2))
    return test_dev, ax

test_dev, ax = deviance_plot(est, X_test, y_test)
ax.legend(loc='upper right')

# <codecell>


