from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import numpy

def run():
    #Adjusted from https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience tutorial
    dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='f8')[1:]
    target = [x[-1] for x in dataset]
    train = [x[0:-1] for x in dataset]
    test = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[0:-1]
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    predicted = rf.predict(test) 
    indices = [x[0] for x in test]
    #print predicted.shape()
    #predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]

    output = numpy.array(zip(indices, predicted), dtype=[('id', int), ('ct',int)])
    savetxt('fake_submission.csv', output, delimiter=',', fmt='%d,%d', 
            header='Id,Cover_Type', comments = '')
    return output

if __name__=="__main__":
  run()
