import numpy as np
from numpy import genfromtxt, savetxt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import decomposition


def load_training_data():
  #loads training data returns as data, target
  dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='f8')[1:]
  data = np.asarray([x[0:-1] for x in dataset]) #remove target
  target = np.asarray([x[-1] for x in dataset])
  return data, target

def load_test_data():
  test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]
  return test

def rf_cross_validate(n=100, X=None, y=None):
  #Cross validation form from http://scikit-learn.org/stable/modules/cross_validation.html
  #Random Forest adjusted from https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience tutorial
  if X==None or y == None:
    X, y = load_training_data()
    X = [x[1:] for x in X] #remove id
  #dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='f8')[1:]
  

  train_x, test_x, train_y, test_y =  cross_validation.train_test_split(X, y, test_size = 0.3)
  rf = RandomForestClassifier(n)
  rf.fit(train_x, train_y)
  return rf.score(test_x, test_y) 

def rf_test_data(n=100, X=None, y=None, test=None, output_file="submission.csv"):
  #Adjusted from https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience tutorial
  if X==None or y == None:
    X, y = load_training_data()
    X = [x[1:] for x in X] #remove id
  #dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='f8')[1:]
  #target = [x[-1] for x in dataset]
  #train = [x[1:-1] for x in dataset]
  #print len(train[0])
  #print target
  if test == None:
    test = load_test_data()
    
  indices = [x[0] for x in test]
  test = [x[1:] for x in test]

  rf = RandomForestClassifier(n_estimators=100)
  rf.fit(X, y)
  predicted = rf.predict(test) 
  output = np.array(zip(indices, predicted), dtype=[('id', int), ('ct',int)])
  savetxt(output_file, output, delimiter=',', fmt='%d,%d', 
          header='Id,Cover_Type', comments = '')
  return output

def pca_explore(n_components=3, test=False):
  #adapted from http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
  #returns X, y if test=False
  #  if test=True returns X, y, T (transformed training data, target, transformed test data)
  np.random.seed(5)
  X, y = load_training_data()
  X = np.asarray([x[1:] for x in X])
  y = np.asarray(y)

  pca = decomposition.PCA(n_components)
  pca.fit(X)
  X_new = pca.transform(X)

  if test:
    #in this case returned y not of use
    test = load_test_data()
    indices = np.reshape(np.asarray([x[0] for x in test]), (len(test), 1))
    T = pca.transform([x[1:] for x in test])
    T = np.append(indices, T, 1)
    return X_new, y, T
  
  return X_new, y
  #return pca

def plot_pca(X=None, y=None):
  fig = plt.figure(1, figsize=(4, 3))
  plt.clf()
  ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
  plt.cla()

  for name, label in [('type 1', 1), ('type 2', 2), ('type 3', 3)]:
      ax.text3D(X[y == label, 0].mean(),
                X[y == label, 1].mean() + 1.5,
                X[y == label, 2].mean(), name,
                horizontalalignment='center',
                bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
  # Reorder the labels to have colors matching the cluster results
  #y = np.choose(y, [1, 2, 3]).astype(np.int)
  ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

  x_surf = [X[:, 0].min(), X[:, 0].max(),
            X[:, 0].min(), X[:, 0].max()]
  y_surf = [X[:, 0].max(), X[:, 0].max(),
            X[:, 0].min(), X[:, 0].min()]
  x_surf = np.array(x_surf)
  y_surf = np.array(y_surf)
  v0 = pca.transform(pca.components_[0])
  v0 /= v0[-1]
  v1 = pca.transform(pca.components_[1])
  v1 /= v1[-1]

  ax.w_xaxis.set_ticklabels([])
  ax.w_yaxis.set_ticklabels([])
  ax.w_zaxis.set_ticklabels([])

  plt.show()

if __name__=="__main__":
  run()
