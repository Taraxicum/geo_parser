import numpy as np
from numpy import genfromtxt, savetxt

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import decomposition

import operator
import random

##########Data Input/Output/Transformation functions######################
def transform_data(test=False):
  #Function used to create condensed training data.  Currently set to just return so as to not accidentally write to the file that has already been finished
  return False
  if test:
    out_file = "test_condense_wild_soil.csv"
    X = load_test_data()
  else:
    out_file = "train_condense_wild_soil.csv"
    X, y = load_training_data()
  #11 - 14 are wilderness area indicators
  wilderness = []
  soil = []
  for r, x in enumerate(X):
    soil.append(0) #should be filled by end but just in case there are records with no soil types defined.
    if x[11] == 1:
      wilderness.append(1)
    elif x[12] == 1:
      wilderness.append(2)
    elif x[13] == 1:
      wilderness.append(3)
    elif x[14] == 1:
      wilderness.append(4)
    else:
      wilderness.append(0)
    for i in range(15,55):
      if x[i] == 1:
        soil[r] = i -14
        break
  with open(out_file, "a") as myfile:
    for r, x in enumerate(X):
      if test:
        output = np.concatenate((x[0:11], [wilderness[r], soil[r]]))
      else:
        output = np.concatenate((x[0:11], [wilderness[r], soil[r], y[r]]))
      myfile.write(",".join(str(int(o)) for o in output) + "\n")

def bootstrap_training_data(X=None, y=None, size=15120, remove_target=True):
  #returns bootstrapped sample of approximately the given size with
  #distribution of wilderness areas similar to the test data

  target_proportions = [.45, .05, .44, .06] #measured directly from test data
  target_size = [int(x*size) for x in target_proportions]

  if X == None or y == None:
    X, y = load_training_data(True, False)
  xpart = partition_on_wilderness(X) #note that the partitions will be indexed 1, 2, 3, 4 rather than starting at 0
  part_sizes = [len(x) for x in xpart]
  new_sample = []
  random.seed(5)
  for i in range(0, 4):
    for j in range(0, target_size[i]):
      rand_val = random.randint(0, part_sizes[i+1] - 1) 
      new_sample.append(xpart[i+1][rand_val])
  return separate_target(new_sample, remove_target)

def separate_target(X, remove_target=True):
  #loads training data returns as data, target.  If remove_target, it removes the target field from the data set and it will only be available in the target array
  if remove_target:
    data = np.asarray([x[0:-1] for x in X]) #remove target
  else:
    data = np.asarray(X)
  target = np.asarray([x[-1] for x in X])
  return data, target


def load_training_data(condensed=True, remove_target=True):
  if condensed:
    dataset = genfromtxt(open('train_condense_wild_soil.csv', 'r'), delimiter=',', dtype='f8')[1:]
  else:
    dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='f8')[1:]
  return separate_target(dataset, remove_target)

def load_test_data(condensed=False):
  if condensed:
    test = genfromtxt(open('test_condense_wild_soil.csv','r'), delimiter=',', dtype='f8')[1:]
  else:
    test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]
  return test

def remove_wilderness_area(data):
  #assumes forest cover data set without id included that has been condensed (i.e. wilderness areas combined to single field and soil types to single field
  #in regards to this data that means the wilderness area field should be at index 10
  #returns array with wilderness area field removed
  #this is to test the hypothesis that wilderness areas in the training data may have over trained some cover types since for example, cover type 4 seems to be associated only with one of the wilderness areas in the training data, but it may be that it does exist in the other wilderness areas
  return [np.concatenate([x[0:10],x[11:]],0) for x in data]

def merge_by_prob(file1, file2, output_file):
  #format of input files should be id,ct,max_p, etc.
  #Output will be id, ct where ct is the cover type with the max_p of the two models being compared
  data1 = genfromtxt(open(file1,'r'), delimiter=',', dtype='f8')[1:]
  data2 = genfromtxt(open(file2,'r'), delimiter=',', dtype='f8')[1:]
  merged = [[x[0], max_ct(x, data2[i])] for i, x in enumerate(data1)]
  savetxt(output_file, merged, delimiter=',', fmt='%d,%d', 
        header='Id,Cover_Type', comments = '')

def max_ct(d1, d2):
  if d1[2] < d2[2]:
    return d2[1]
  else:
    return d1[1]

##########Machine Learning Functions################################
def test_parameters(n_tests, parameter, p_range, keywords={}):
  results = []
  X, y = load_training_data()
  X = [x[1:] for x in X]
  for i, p in enumerate(p_range):
    results.append([])
    p_keywords = keywords
    p_keywords[parameter] = p
    for j in range(0, n_tests):
      results[i].append(knn_cross_validate(X=X, y=y, keywords=p_keywords, random_state=j, just_score=True))
  return results

def cross_validate(classifier_type, args=(), keywords={}, test_size=0.3, X=None, y=None, random_state=5, just_score=False):
  if X==None or y == None:
    X, y = load_training_data()
    X = [x[1:] for x in X] #remove id
  train_x, test_x, train_y, test_y =  cross_validation.train_test_split(X, y, test_size=test_size, random_state=random_state)
  classifier = classifier_type(*args, **keywords)
  classifier.fit(train_x, train_y)
  if just_score:
    return classifier.score(test_x, test_y)
  else:
    return classifier.score(test_x, test_y), classifier

def rf_cross_validate(n=100, X=None, y=None):
  #Cross validation form from http://scikit-learn.org/stable/modules/cross_validation.html
  #Random Forest adjusted from https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience tutorial
  return cross_validate(RandomForestClassifier, (n,), keywords={'random_state':5}, test_size=.1, X=X, y=y)

def knn_cross_validate(X=None, y=None, keywords={"n_neighbors":5, "weights":"distance",'p':1}, test_size=.3, random_state=5, just_score=False):
  return cross_validate(KNeighborsClassifier, keywords=keywords, X=X, y=y, test_size=test_size, just_score=just_score, random_state=random_state)



def partition_rf(n=100, output_file="partition_rf.csv"):
  X, y = load_training_data(True, False)
  test = load_test_data(True)

  train_part = partition_on_wilderness(X)
  test_part = partition_on_wilderness(test)

  x_part = []
  y_part = []
  indices = []
  t_part = []
  for i in range(1, 5):
    x, y = separate_target(train_part[i])
    x_part.append([xv[1:] for xv in x]) #remove id field
    y_part.append(y)
    indices.append([x[0] for x in test_part[i]])
    t_part.append([x[1:] for x in test_part[i]])

  output = []
  for i in range(0, 4):
    r = RandomForestClassifier(n, random_state=5)
    r.fit(x_part[i], y_part[i])
    predicted = r.predict(t_part[i]) 
    output.append(np.array(zip(indices[i], predicted), dtype=[('id', int), ('ct',int)]))
  true_out = []
  for i in range(0,4):
    for item in output[i]:
      true_out.append(item)
  savetxt(output_file, true_out, delimiter=',', fmt='%d,%d', 
        header='Id,Cover_Type', comments = '')

def rf_test_data(n=100, X=None, y=None, test=None, output_file="submission.csv", probabilities=False, both=False):
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
  rf = RandomForestClassifier(n_estimators=n, random_state=5)
  rf.fit(X, y)
  if probabilities or both:
    predicted = rf.predict_proba(test)
    max_p = [max(enumerate(x), key = operator.itemgetter(1)) for x in predicted]
    output = [np.concatenate(([x[0]], [x[1][0]+1, x[1][1]], x[2]), 0) for x in zip(indices, max_p, predicted)]

    #output = np.array(zip(indices[0:10], predicted[0:10]), dtype=[('id', int), [('c1',int), ('c2',int), ('c3',int), ('c4',int), ('c5',int), ('c6',int), ('c7',int)]])
    p_outfile = "proba_{}".format(output_file)
    savetxt(p_outfile, output, delimiter=',', fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f', 
          header='Id,CT,p,Ct1,Ct2,Ct3,Ct4,Ct5,Ct6,Ct7', comments = '')
  if both or not probabilities:
    #output_file = "submission.csv" #not same as probabilities file
    predicted = rf.predict(test) 
    output = np.array(zip(indices, predicted), dtype=[('id', int), ('ct',int)])
    savetxt(output_file, output, delimiter=',', fmt='%d,%d', 
          header='Id,Cover_Type', comments = '')
  return True#output

def pca_explore(n_components=3, test=False):
  #adapted from http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
  #returns X, y if test=False
  #  if test=True returns X, y, T (transformed training data, target, transformed test data)
  np.random.seed(50)
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

def pca_to_rf(n_components=5, n_trees=100, test=False):
  if test:
    X, y, T = pca_explore(n_components, test)
    out_file="PCA{}_RF{}.csv".format(n_components, n_trees)
    result = "Not yet known(see kaggle)"
  else:
    X, y = pca_explore(n_components)
    result, rf = rf_cross_validate(n=n_trees, X=X, y=y)
    out_file = "NA"
    process = "PCA_{} -> RF_{}".format(n_components, n_trees)
  return result, out_file, process

def condensed_rf(n_trees=100, test=False):
  if test:
    X, y = load_training_data()
    X_new = [x[1:] for x in X]
    T = load_test_data(condensed=True)
    result = "Not yet known(see kaggle)"
    out_file="condensed_submission_RF{}.csv".format(n_trees)
    rf_test_data(n=n_trees, X=X_new, y=y, test=T, output_file=out_file)
  else:
    X, y = load_training_data()
    result, rf = rf_cross_validate(n=n_trees, X=X, y=y)
    out_file = "NA"
  process = "condense_wild_soil RF{}".format(n_trees)
  return result, out_file, process

def bootstrapped_rf(n_trees=100, test=False):
  if test:
    X, y = bootstrap_training_data()
    X_new = [x[1:] for x in X]
    T = load_test_data(condensed=True)
    result = "Not yet known(see kaggle)"
    out_file="bootstrap_wilderness_distribution_RF{}.csv".format(n_trees)
    rf_test_data(n=n_trees, X=X_new, y=y, test=T, output_file=out_file)
  else:
    X, y = bootstrap_training_data()
    result, rf = rf_cross_validate(n=n_trees, X=X, y=y)
    out_file = "NA"
  process = "bootstrap: wilderness distribution RF{}".format(n_trees)
  return result, out_file, process

def partitioned_rf(n_trees=100):
  out_file="partitioned_wilderness_RF{}.csv".format(n_trees)
  partition_rf(n_trees, out_file)
  result = "Not yet known(see kaggle)"
  process = "partitioned on wilderness RF{}".format(n_trees)
  return result, out_file, process
  

##########helper function for keeping track of results
def keep_track(func, args):
  import time
  from time import strftime
  RESULTS = "results.csv"
  date = strftime("%Y-%m-%d %H:%M:%S")
  start_time = time.time()
  result, out_file, process = func(*args)
  end_time = time.time()
  speed = "{:.1f}".format(end_time - start_time)
  print date + "," + str(result) + "," + speed +"," + out_file + "," + process
  with open(RESULTS, "a") as myfile:
    myfile.write("\n" + date + "," + str(result) + "," + speed +"," + out_file + "," + process)



###########Functions for display of data############################

def partition_on_wilderness(X):
  xpartition = []
  for i in range(0, 5):
    xpartition.append([])
  for x in X:
    xpartition[int(x[11])].append(x)
  return xpartition


def prepare_for_scatter_on_cover(X, y):
  xpartition = []
  for i in range(0, 8):
    xpartition.append([])
  for r, v in enumerate(y):
    xpartition[int(v)].append(X[r])
  return xpartition

def fudge(i, m):
  return 0#(i - m/4+ (np.random.rand() - .5)/2)/4

def plot_scatter_by_cover_filter_by_wilderness(train, test, xind, yind, wilderness):
  colors = ["b.", "g.", "r.", "c.", "m.", "y.", "k."]
  w_labels = ["Rawah", "Neota", "Comanche Peak", "Cache La Poudra"]
  labels = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
  fields = ["id", "Elevation", "Aspect", "Slope", "Horizontal distance to water", "Vertical distance to water", "Horizontal distance to roadway", "Hillshade 9am", "Hillshade noon", "Hillshade 3pm", "Horizontal distance to fire points", "Wilderness Area", "Soil Type", "Cover Type"]
  fig = plt.figure()
  ax = fig.add_subplot(121)
  ax.set_title("training")
  for i in range(1, 8):
    x_vals = [x[xind] + fudge(i, 7) for x in train[i] if int(x[11]) == wilderness]
    y_vals = [y[yind] for y in train[i] if int(y[11])==wilderness]
    plt.plot(x_vals, y_vals, colors[i-1], label=labels[i-1], markersize=1.2)
  plt.legend(framealpha=.5, markerscale=7)
  ax = fig.add_subplot(122)
  ax.set_title("test")
  x_vals = [x[xind] for x in test if int(x[11]) == wilderness]
  y_vals = [y[yind] for y in test if int(y[11])==wilderness]
  plt.plot(x_vals, y_vals, "b.",label=w_labels[wilderness-1], markersize=1.2)
  plt.legend(framealpha=.5, markerscale=7)
  title = "{} vs {}".format(fields[xind], fields[yind])
  plt.title(title)

def plot_scatter_by_wilderness(train_partition, test_partition, xind, yind, title):
  colors = ["b.", "m.", "r.", "k."]
  labels = ["Rawah", "Neota", "Comanche Peak", "Cache La Poudra"]
  fields = ["id", "Elevation", "Aspect", "Slope", "Horizontal distance to water", "Vertical distance to water", "Horizontal distance to roadway", "Hillshade 9am", "Hillshade noon", "Hillshade 3pm", "Horizontal distance to fire points", "Wilderness Area", "Soil Type", "Cover Type"]
  if title == "" or title == None:
    title = "{} vs {}".format(fields[xind], fields[yind])
  plt.title(title)
  fig = plt.figure()
  ax = fig.add_subplot(121)
  ax.set_title("training")
  for i in range(1, 5):
    plt.plot([x[xind] + fudge(i, 4) for x in train_partition[i]], [y[yind] for y in train_partition[i]],colors[i-1], label=labels[i-1], markersize=1.2)
  ax2 = fig.add_subplot(122)
  ax2.set_title("test")
  for i in range(1, 5):
    plt.plot([x[xind] + fudge(i, 4) for x in test_partition[i]], [y[yind] for y in test_partition[i]],colors[i-1], label=labels[i-1], markersize=1.2)

  plt.legend(framealpha=.5, markerscale=7)
  plt.show()

def plot_scatter_by_cover(xpartition, xind, yind, title):
  colors = ["b.", "g.", "r.", "c.", "m.", "y.", "k."]
  labels = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
  fields = ["id", "Elevation", "Aspect", "Slope", "Horizontal distance to water", "Vertical distance to water", "Horizontal distance to roadway", "Hillshade 9am", "Hillshade noon", "Hillshade 3pm", "Horizontal distance to fire points", "Wilderness Area", "Soil Type", "Cover Type"]
  if title == "" or title == None:
    title = "{} vs {}".format(fields[xind], fields[yind])
  for i in range(1, 8):
    plt.plot([x[xind] + fudge(i, 7) for x in xpartition[i]], [y[yind] for y in xpartition[i]],colors[i-1], label=labels[i-1], markersize=1.2)
    plt.title(title)
    plt.legend(framealpha=.5, markerscale=7)
  plt.show()

def plot_histograms(x, xedge, y, title):
  xedges = xedge
  yedges = (1, 2, 3, 4, 5, 6, 7, 8)
  H, xedges, yedges = np.histogram2d(x, y, [xedges, yedges])
  fig = plt.figure()
  #ax = fig.add_subplot(111)
  #ax.set_title("imshow equidistant")
  #im = plt.imshow(H, interpolation='none', origin='low')
  #ax.set_xlim(xedges[0], xedges[-1])
  #ax.set_ylim(yedges[0], yedges[-1])
  
  ax = fig.add_subplot(111)
  ax.set_title(title)
  X, Y = np.meshgrid(yedges, xedges)
  plot1 = ax.pcolormesh(X, Y, H)
  #ax.set_aspect('equal')
  plt.colorbar(plot1)
  plt.show()



if __name__=="__main__":
  run()
