import numpy as np
from numpy import genfromtxt, savetxt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def cross_validate():
  #Cross validation form from http://scikit-learn.org/stable/modules/cross_validation.html
  #seems to be working with this data: result ~.86
  dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='f8')[1:]
  
  data = [x[1:-1] for x in dataset]
  target = [x[-1] for x in dataset]
  train_x, test_x, train_y, test_y =  cross_validation.train_test_split(data, target, test_size = 0.3)
  rf = RandomForestClassifier(n_estimators=100)
  rf.fit(train_x, train_y)
  return rf.score(test_x, test_y) 

def test_data():
  #Adjusted from https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience tutorial
  dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='f8')[1:]
  target = [x[-1] for x in dataset]
  train = [x[1:-1] for x in dataset]
  #print len(train[0])
  #print target
  test_dataset = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]
  test = [x[1:] for x in test_dataset]
  indices = [x[0] for x in test_dataset]

  rf = RandomForestClassifier(n_estimators=100)
  rf.fit(train, target)
  predicted = rf.predict(test) 
  output = np.array(zip(indices, predicted), dtype=[('id', int), ('ct',int)])
  savetxt('submission.csv', output, delimiter=',', fmt='%d,%d', 
          header='Id,Cover_Type', comments = '')
  return output

def run():
  #Adjusted from https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience tutorial
  #Cross validation form from http://scikit-learn.org/stable/modules/cross_validation.html
  dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='f8')[1:]
  
  data = [x[1:-1] for x in dataset]
  target = [x[-1] for x in dataset]
  train_x, test_x, train_y, test_y =  cross_validation.train_test_split(data, target, test_size = 0.3)
  #target = [x[-1] for x in dataset]
  #train = [x[0:-1] for x in dataset]
  #test = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[0:-1]
  rf = RandomForestClassifier(n_estimators=100)
  rf.fit(train_x, train_y)
  #rf.fit(train[1:-1], train[-1])
  return rf.score(test_x, test_y) 
  #predicted = rf.predict(test) 
  #indices = [x[0] for x in test]
  #print predicted.shape()
  #predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]

  #output = np.array(zip(indices, predicted), dtype=[('id', int), ('ct',int)])
  #savetxt('fake_submission.csv', output, delimiter=',', fmt='%d,%d', 
          #header='Id,Cover_Type', comments = '')
  #return output

if __name__=="__main__":
  run()
