import numpy as np
from numpy import genfromtxt, savetxt

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC

import operator
import random


class ForestCoverData():
  ##########Constants/Labels################################################
    COVER = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
    FIELDS = ["id", "Elevation", "Aspect", "Slope", "Horizontal distance to water", "Vertical distance to water", "Horizontal distance to roadway", "Hillshade 9am", "Hillshade noon", "Hillshade 3pm", "Horizontal distance to fire points", "Wilderness Area", "Soil Type", "Cover Type"]
    WILDERNESS = ["Rawah", "Neota", "Comanche Peak", "Cache La Poudra"]

  ##########Data Input/Output/Transformation functions######################
    def __init__(self, condensed=True):
      self.condensed = condensed
      self.load_training_data()
      self.load_test_data()
      self.normalized = None

    def add_field(self, X, field):
      return [np.append(X[i], field[i]) for i, v in enumerate(X)]
    
    def limit_cover_types(self, types=None):
      if types == None:
        types = [1, 2]
      new_X = []
      new_y = []
      for i, v in enumerate(self.y):
        if v in types:
          new_X.append(self.X[i])
          new_y.append(v)
      return new_X, new_y
    
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

    def separate_target(self, X, remove_target=True):
      #loads training data returns as data, target.  If remove_target, it removes the target field from the data set and it will only be available in the target array
      if remove_target:
        data = np.asarray([x[0:-1] for x in X]) #remove target
      else:
        data = np.asarray(X)
      target = np.asarray([x[-1] for x in X])
      return data, target

    def load_probability_dataset(self, infile):
      dataset = genfromtxt(open(infile, 'r'), delimiter=',', dtype='f8')[1:]
      return dataset


    def load_training_data(self, remove_target=True):
      if self.condensed:
        dataset = genfromtxt(open('train_condense_wild_soil.csv', 'r'), delimiter=',', dtype='f8')[1:]
      else:
        dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='f8')[1:]
      self.X, self.y = self.separate_target(dataset, remove_target)
      self.X_no_id = [x[1:] for x in self.X]

    def load_test_data(self):
      if self.condensed:
        self.test = genfromtxt(open('test_condense_wild_soil.csv','r'), delimiter=',', dtype='f8')[1:]
      else:
        self.test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]
      self.test_no_id = [x[1:] for x in self.test]

    def remove_wilderness_area(self):
      #assumes forest cover data set without id included that has been condensed (i.e. wilderness areas combined to single field and soil types to single field
      #in regards to this data that means the wilderness area field should be at index 10
      #returns array with wilderness area field removed
      #this is to test the hypothesis that wilderness areas in the training data may have over trained some cover types since for example, cover type 4 seems to be associated only with one of the wilderness areas in the training data, but it may be that it does exist in the other wilderness areas
      return [np.concatenate([x[0:10],x[11:]],0) for x in self.X_no_id]

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

  ###########Functions for display of data############################
    def plot_1d_histograms(self, X = None, test = None, save_root=None):
      if X == None:
        X = self.X
      if test == None:
        test = self.test
      if save_root == None:
        save_root = "autographs/1d_histogram_{}.png"
      
      for i, v in enumerate(self.FIELDS[0:-1]):
        savepath =save_root.format(v)
        max_v = max(max([x[i] for x in X]), max([x[i] for x in test]))
        min_v = min(min([x[i] for x in X]), min([x[i] for x in test]))
        if i == 11: #Wilderness Area
          bins = range(0, 6)
        elif i == 12: #soil type
          bins = range(1, 42)
        else:
          bins = range(int(min_v), int(max_v), int((max_v - min_v)/10))
        fig = plt.figure()
        plt.suptitle("1d histogram of {}".format(v))
        ax = fig.add_subplot(211)
        ax.set_title("Training Data")
        ax.set_xlim(min_v, max_v + 1) 
        if i==11:
          plt.xticks([1, 2, 3, 4, 5])
        plt.hist([x[i] for x in X], bins=bins)
        ax = fig.add_subplot(212)
        ax.set_title("Test Data")
        ax.set_xlim(min_v, max_v) 
        plt.hist([x[i] for x in test], bins=bins)
        if i==11:
          plt.xticks([1, 2, 3, 4, 5])
        plt.savefig(savepath)
        plt.clf()

    def plot_histograms_vs_cover(self):
      for i in range(1, 11):
        self.plot_histogram_vs_cover(i)
      self.plot_histogram_vs_cover(11, xedges=range(1, 6))
      self.plot_histogram_vs_cover(12, xedges=range(1, 42))

    def plot_histogram_vs_cover(self, xind, xedges=None):
      title = "Cover type vs. {} (training data)".format(self.FIELDS[xind])
      savepath = "autographs/histogram_cover_vs_{}.png".format(self.FIELDS[xind])
      x = [int(v[xind]) for v in self.X]
      min_x = int(min(x))
      max_x = int(max(x))
      if xedges == None:
        xedges = range(min_x, max_x+1, int((max_x-min_x)/20))
      yedges = (1, 2, 3, 4, 5, 6, 7, 8)
      H, xedges, yedges = np.histogram2d(x, self.y, [xedges, yedges])
      fig = plt.figure()
      
      ax = fig.add_subplot(111)
      ax.set_title(title)
      X, Y = np.meshgrid(yedges, xedges)
      plot1 = ax.pcolormesh(X, Y, H)
      plot1.axes.set_ylim(min_x, max(xedges)) 
      plt.colorbar(plot1)
      if xind==11:
        plt.yticks([1, 2, 3, 4, 5])
      plt.savefig(savepath)

    def partition_on_wilderness(X):
      xpartition = []
      for i in range(0, 5):
        xpartition.append([])
      for x in X:
        xpartition[int(x[11])].append(x)
      return xpartition

    def filter_by_index_partition_on_cover(filter_set, cover_id):
      test = load_test_data()
      xpart = []
      for i in range(1, 8):
        xpart.append([])
      for v in filter_set:
        xpart[int(v[cover_id]-1)].append(test[int(v[0])])
      return xpart

    def partition_on_cover(self, X=None, y=None):
      if X == None or y == None:
        X = self.X
        y = self.y

      xpartition = []
      for i in range(0, len(set(y))+1):
        xpartition.append([])
      for r, v in enumerate(y):
        xpartition[int(v)].append(X[r])
      return xpartition

    def fudge(i, m):
      return 0#(i - m/4+ (np.random.rand() - .5)/2)/4

    def plot_scatter_clustering(self, data, xind, yind):
      colors = ["#000000", "#330000", "#003300", "#000033",
                "#333300", "#330033", "#003333", "#660000",
                "#009900", "#0000ee", "#888888", "#ffffff"]
      colors = ["b.", "g.", "r.", "c.", "m.", "y.", "k.", "o."]
      for i in range(0, len(data)):
        x_vals = [x[xind] for x in data[i]]
        y_vals = [x[yind] for x in data[i]]
        print len(x_vals)
        plt.plot(x_vals, y_vals, colors[i], markersize=1.2)

    def plot_scatter_by_cover_filter_by_wilderness(train, test, xind, yind, wilderness):
      #training set should be partitioned by cover type
      colors = ["b.", "g.", "r.", "c.", "m.", "y.", "k.", "o."]
      fig = plt.figure()
      ax = fig.add_subplot(121)
      ax.set_title("training")
      
      #Plot training set data
      for i in range(1, 8):
        x_vals = [x[xind] + fudge(i, 7) for x in train[i] if int(x[11]) == wilderness]
        y_vals = [y[yind] for y in train[i] if int(y[11])==wilderness]
        plt.plot(x_vals, y_vals, colors[i-1], label=COVER[i-1], markersize=1.2)
      plt.legend(framealpha=.5, markerscale=7)
      ax = fig.add_subplot(122)
      ax.set_title("test")
      
      #Plot test set data
      x_vals = [x[xind] for x in test if int(x[11]) == wilderness]
      y_vals = [y[yind] for y in test if int(y[11])==wilderness]
      plt.plot(x_vals, y_vals, "b.",label=WILDERNESS[wilderness-1], markersize=1.2)
      plt.legend(framealpha=.5, markerscale=7)
      title = "{} vs {}".format(FIELDS[xind], FIELDS[yind])
      plt.title(title)

    def plot_scatter_by_wilderness(train_partition, test_partition, xind, yind):
      #training and test sets should be partitioned by wilderness area for use in this function
      colors = ["b.", "m.", "r.", "k."]
      title = "{} vs {}".format(FIELDS[xind], FIELDS[yind]) #TODO need to figure out how to get this main title as well as titles for the subplots
      fig = plt.figure()
      ax = fig.add_subplot(121)
      ax.set_title("training")
      for i in range(1, 5):
        plt.plot([x[xind] + fudge(i, 4) for x in train_partition[i]], [y[yind] for y in train_partition[i]],colors[i-1], label=WILDERNESS[i-1], markersize=1.2)
      ax2 = fig.add_subplot(122)
      ax2.set_title("test")
      for i in range(1, 5):
        plt.plot([x[xind] + fudge(i, 4) for x in test_partition[i]], [y[yind] for y in test_partition[i]],colors[i-1], label=WILDERNESS[i-1], markersize=1.2)

      plt.legend(framealpha=.5, markerscale=7)
      plt.show()

    def plot_train_vs_test(self, xind, yind, X=None, y=None):
      self.plot_records(xind, yind, self.test)

      if X==None or y==None:
        X = self.X
        y = self.y

      self.plot_by_cover(X, y, xind, yind)

    
    def plot_by_cover(self, X, y, xind, yind, offset=0):
      colors = ["b.", "r.", "g.", "c.", "m.", "y.", "k."]
      title = "{} vs {}".format(self.FIELDS[xind], self.FIELDS[yind])
      xind += offset
      yind += offset
      
      for ct in set(y):
        plt.plot([X[i][xind] for i, v in enumerate(y) if v == ct], [X[i][yind] for i, v in enumerate(y) if v == ct],colors[int(ct-1)], label=self.COVER[int(ct-1)], markersize=4)
        plt.title(title)
        plt.legend(framealpha=.5, markerscale=7)
      plt.show()


    
    def plot_scatter_by_cover(xpartition, xind, yind, title=""):
      colors = ["b.", "r.", "g.", "c.", "m.", "y.", "k."]
      if title == "" or title == None:
        title = "{} vs {}".format(FIELDS[xind], FIELDS[yind])
      for i in range(1, 8):
        plt.plot([x[xind] + fudge(i, 7) for x in xpartition[i]], [y[yind] for y in xpartition[i]],colors[i-1], label=COVER[i-1], markersize=1.2)
        plt.title(title)
        plt.legend(framealpha=.5, markerscale=7)
      plt.show()

  ##########Binary Classifiers########################################
    def combined(self):
      import time
      from time import strftime
      date = strftime("%Y-%m-%d %H:%M:%S")
      start_time = time.time()
      print start_time
      
      values = self.train_on_partitions(for_lp=True)
      self.lp_remaining(values)
      end_time = time.time()
      speed = "{:.1f}".format(end_time - start_time)
      print "end_time {}; speed {}".format(end_time, speed)
    
    def train_on_partitions(self, n=500, for_lp=False):
      #for_lp format appropriate for label propagation/spreading classifier
      ct_vals = set(self.y)
      classifiers = []
      predictions = []

      for v in ct_vals:
        print "starting {}_classifier".format(v)
        target = []
        for i in self.y:
          if i == v:
            target.append(1)
          else:
            target.append(0)
        classifiers.append(RandomForestClassifier(n, random_state=5))
        classifiers[-1].fit(self.X_no_id, target)
        predictions.append(classifiers[-1].predict(self.test_no_id))
      
      #output.append(np.array(zip(indices[i], predicted), dtype=[('id', int), ('ct',int)]))
      if for_lp:
        output = []
      else:
        no_class = []
        multi_class = []
        output = []
      
      for i, v in enumerate(self.test):
        s = 0
        which = [] 
        for j, p in enumerate(predictions):
          s += p[i]
          if p[i] > 0:
            which.append(j+1)
        if for_lp:
          if s == 1:
            output.append([v, which[0]])
          else:
            output.append([v, -1])
        else:
          if s == 0:
            no_class.append(v)
          elif s > 1:
            multi_class.append([v, which])
          else:
            #if which >= 0:
            output.append([v, which[0]])
      if for_lp:
        return output
      else:
        return no_class, multi_class, output

    def lp_remaining(self, values):
      #Run label propagation/spreading algorithm on remaining values from binary classifier scheme
      n = 5
      a = .00001
      lp = LabelSpreading('knn', n_neighbors=n, alpha=a)
      X = [x[0] for x in values]
      y = [x[1] for x in values]
      
      indices = [x[0] for x in self.test]
      X = [x[1:] for x in X]
      lp.fit(X, y)
      output = lp.predict(X)
      
      output = np.array(zip(indices, output), dtype=[('id', int), ('ct',int)])
      print "Sanity check: output length: {}".format(len(output))
      savetxt("binary_rf500_lp_knn{}_a{}.csv".format(n, a), output, delimiter=',', fmt='%d,%d', 
            header='Id,Cover_Type', comments = '')
       

    def classify_remaining(self, no, multi, solid):
      test = np.append(no, [x[0] for x in multi], axis=0)
      indices = [x[0] for x in self.test]
      test_lim = test[...,(1, 4, 5, 6, 10)] #the fields I am interested in clustering on
      print test_lim[0]
      #test = [x[1:] for x in test]
      X = np.asarray([x[0] for x in solid])
      X_ind = [x[0] for x in X]
      #X = [x[1:] for x in X]
      X_lim = X[...,(1, 4, 5, 6, 10)] #the fields I am interested in clustering on
      print X_lim[0]
      y = [x[1] for x in solid]
      
      s, knn = self.knn_cross_validate(X_lim, y, keywords={"n_neighbors":3, "weights":"distance", "p":1})
      predicted = knn.predict(test_lim)
      full_ind = np.append(indices, X_ind)
      full_ct = np.append(predicted, y)
      output = np.array(zip(full_ind, full_ct), dtype=[('id', int), ('ct',int)])
      print "Sanity check: output length: {}".format(len(output))
      savetxt("binary_rf500_lim_knn3.csv", output, delimiter=',', fmt='%d,%d', 
            header='Id,Cover_Type', comments = '')

  ##########Clustering################################################
    def kmeans(X=None, y=None, keywords={'n_clusters':7, 'init':'random', 'n_init':50, 'random_state':5}):
      if X==None:
        X, y = load_training_data(False)
        X = np.asarray([x[1:] for x in X]) #remove id
      k = KMeans(**keywords).fit(X)
      return k

    def dbscan(self, X=None, y=None, keywords={'eps':9.0, 'min_samples':10}):
      if X==None:
        X, y = load_training_data(False)
        X = np.asarray([x[1:] for x in X]) #remove id
      db = DBSCAN(**keywords).fit(X)
      return db

    def partition_by_dbscan(self, X, db):
      xpart = []
      for x in set(db.labels_):
        xpart.append([])
      for i, v in enumerate(db.labels_):
        xpart[int(v+1)].append(X[i])
      return xpart

  ##########Geographical Coordinates##################################
    def prep_coordinates(self, record, threshhold=1, X=None, test=False):
      #find and return cohort of points that are plausibly from
      #  a nearby physical location to the input record
      if X==None:
        if test:
          X = load_test_data(True)
        else:
          X, y = load_training_data(True)
      
      initial = X[record]
      cohort = [x for x in X if initial[11] == x[11] and abs((initial[1] - initial[5]) - (x[1]-x[5])) < threshhold]
      #field 11: check for same wilderness area
      #field 1 - 5: check if difference in elevation and v. distance to water is within threshhold - i.e.
      #  elevation of the nearest body of water is approximately the same for both records
      return cohort

    def split_cohorts(self, seed=150):
      #assumes working with test data
      #print len(self.X)
      #if X == None:
        #X = load_test_data(True)
      #X = np.asarray(X)
      X = self.X
      np.random.seed(seed)
      r = np.random.randint(0, len(X))
      #print "starting points index: {}".format(r)
      cohort = self.prep_coordinates(r, X=X, test=True)
      cohort = np.asarray(cohort)
      #print "Number of points in cohort: {}".format(len(cohort))
      
      #isolate the fields I am interested in clustering on:
      #fields: elevation, horizontal distance to water, vertical distance to water, 
      #        h. distance to road, and h. distance to fire points
      rc = cohort[...,(1, 4, 5, 6, 10)] 
     
      #Further partition cohort by clustering points that are near each other
      db = self.dbscan(rc, keywords={'eps':150.0, 'min_samples':5})
      xpart = self.partition_by_dbscan(cohort, db)

      #show example plot with one of the subcohorts
      self.plots_for_subcohort(xpart[1], cohort, X, True)
      plt.show()
      return xpart


    def plot_records(self, xind, yind, records, wilderness=None, color='c.'):
      if wilderness != None:
        x_vals = [x[xind] for x in records if int(x[11]) == wilderness]
        y_vals = [x[yind] for x in records if int(x[11]) == wilderness]
      else:
        x_vals = [x[xind] for x in records]
        y_vals = [x[yind] for x in records]
      plt.plot(x_vals, y_vals, color, markersize=1.0)

    def plots_for_subcohort(self, subcohort, cohort, X=None, test=False):
      if X==None:
        if test:
          X = load_test_data(True)
        else:
          X, y = load_training_data(True)
      wilderness = int(subcohort[0][11])
      xinds = [4, 5, 6, 10]
      yind = 1
      fig = plt.figure()
      plt.suptitle("{} Wilderness".format(self.WILDERNESS[int(cohort[0][11])]))
      
      for i, xind in enumerate(xinds):
        ax = fig.add_subplot(2,2,i+1)
        title = "{} vs {}".format(self.FIELDS[xind], self.FIELDS[yind])
        plt.title(title)
        self.plot_records(xind, yind, X, wilderness, 'r.')
        self.plot_records(xind, yind, cohort, wilderness, 'bo')
        self.plot_records(xind, yind, subcohort, wilderness, 'g>')

    def plot_cohort_against_background(xind, yind, cohort, X=None, test=False, color='bo'):
      if X==None:
        if test:
          X = load_test_data(True)
        else:
          X, y = load_training_data(True)
      plot_background(xind, yind, int(cohort[0][11]), X)
      plot_cohort(xind, yind, cohort, color)


    def plot_cohort(xind, yind, cohort, color='bo'):
      x_vals = [x[xind] for x in cohort]
      y_vals = [x[yind] for x in cohort]
      
      title = "{} vs {}\n{} Wilderness".format(FIELDS[xind], FIELDS[yind], WILDERNESS[int(cohort[0][11])])
      plt.title(title)
      plt.plot(x_vals, y_vals, color)

    def plot_background(xind, yind, wilderness, X=None):
      if X == None:
        X, y = load_training_data(True)
      x_vals = [x[xind] for x in X]# if int(x[11]) == wilderness]
      y_vals = [x[yind] for x in X]# if int(x[11]) == wilderness]
      
      title = "{} vs {}".format(FIELDS[xind], FIELDS[yind])
      plt.title(title)
      plt.plot(x_vals, y_vals, 'r.', markersize=1.2)

    def plot_right_vs_wrong(self, right, wrong, xind, yind):
      title = "{} vs {}".format(self.FIELDS[xind+1], self.FIELDS[yind+1])
      plt.title(title)
      rx_vals = [x[xind] for x in right]
      ry_vals = [x[yind] for x in right]
      wx_vals = [x[xind] for x in wrong]
      wy_vals = [x[yind] for x in wrong]
      plt.plot(rx_vals, ry_vals, 'b.')
      plt.plot(wx_vals, wy_vals, 'r.')
      plt.show()


  ##########Combining Techniques######################################
    def best_predicted_sets(in_file, threshhold = .8, max_col = 3):
      X = load_probability_dataset(in_file)
      return [x[0:3] for x in X if x[max_col] > threshhold]

  ##########Machine Learning Functions################################
    def test_parameters(self, n, estimator, param_grid, X=None, y=None):
      random_state = 5
      if X==None or y == None:
        X = self.X_no_id
        y = self.y
      
      train_x, test_x, train_y, test_y =  cross_validation.train_test_split(X, y, test_size=.15, random_state=random_state)

      results = []
      cv = cross_validation.ShuffleSplit(len(train_x), n, test_size=.10, random_state=random_state)
      #scores = cross_validation.cross_val_score(classifier, train_X, train_y, cv=cv)
      gs = GridSearchCV(estimator, param_grid, cv=cv)
      return gs.fit(train_x, train_y)

    def plot_confusion(self, diff):
      xvals = [x[0] for x in diff]
      yvals = [x[1] for x in diff]
      H, xedges, yedges = np.histogram2d(xvals, yvals, [range(1, 9), range(1, 9)])
      X, Y = np.meshgrid(xedges, yedges)
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.set_title("Predicted vs. True")
      plot = ax.pcolormesh(X, Y, H)
      plt.colorbar(plot)
      plt.show()

    def limit_cover_types(self, cts):
      ylim = []
      xlim = []
      for i, v in enumerate(self.y):
        if v in cts:
          ylim.append(v)
          xlim.append(self.X[i])
      return xlim, ylim

    
    def explore_wrong_classifications(self, X=None, y=None, classifier_type=RandomForestClassifier, args=(), keywords={}, random_state=5):
      if X == None or y == None:
        X = self.X_no_id
        y = self.y
      test_size=.2
      args = (100, )

      train_x, test_x, train_y, test_y =  cross_validation.train_test_split(X, y, test_size=test_size, random_state=random_state)
      classifier = classifier_type(*args, **keywords)
      classifier.fit(train_x, train_y)
      print classifier.score(test_x, test_y)
      predicted = classifier.predict(test_x)
      correct = []
      incorrect = []
      diff = []
      for i, v in enumerate(predicted):
        if v == test_y[i]:
          correct.append(np.append(test_x[i], v))
          diff.append([v, test_y[i]])
        else:
          incorrect.append(np.append(test_x[i], test_y[i]))
          diff.append([v, test_y[i]])
      return correct, incorrect, diff

    def cross_validate(self, classifier_type, args=(), keywords={}, test_size=0.3, X=None, y=None, random_state=5, just_score=False, n=10):
      classifier = classifier_type(*args, **keywords)
      if just_score:
        #train_x, test_x, train_y, test_y =  cross_validation.train_test_split(X, y, test_size=test_size, random_state=random_state)
        cv = cross_validation.ShuffleSplit(len(X), n, test_size, random_state=random_state)
        scores = cross_validation.cross_val_score(classifier, X, y, cv=cv)
        print scores
        print "n = {}, sd = {:.4f}, mean = {:.4f}".format(n, np.std(scores), np.mean(scores))
        return np.mean(scores)
        #return classifier.score(test_x, test_y)
      else:
        train_x, test_x, train_y, test_y =  cross_validation.train_test_split(X, y, test_size=test_size, random_state=random_state)
        classifier.fit(train_x, train_y)
        return classifier.score(test_x, test_y), classifier

    def label_prop_cv(self, n=2, a=1.0, test_size=0.3, random_state=5, X=None, y=None):
      if X == None or y == None:
        X = self.X_no_id
        y = self.y
      train_x, test_x, train_y, test_y =  cross_validation.train_test_split(X[0:5000], y[0:5000], test_size=test_size, random_state=random_state)
      lp = LabelSpreading('knn', n_neighbors=n, alpha=a)
      X = np.append(train_x, test_x, axis=0)
      #X = X[..., (0, 3, 5, 6, 7, 9, 11)]
      #test_x = test_x[..., (0, 3, 5, 6, 7, 9, 11)]
      y = np.append(train_y, np.zeros(len(test_y))-1)
      lp.fit(X, y)
      return lp.score(test_x, test_y)

    def svm_cross_validate(self, keywords={}):
      if self.normalized == None:
        self.normalized = preprocessing.normalize(self.X_no_id)
      return self.cross_validate(SVC, (), keywords=keywords, test_size=.2, X=self.normalized, y=self.y)
      
    
    def rf_cross_validate(self, n=100, X=None, y=None, keywords={'random_state':5}):
      #Cross validation form from http://scikit-learn.org/stable/modules/cross_validation.html
      #Random Forest adjusted from https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience tutorial
      if X == None or y == None:
        X = self.X_no_id
        y = self.y
      return self.cross_validate(RandomForestClassifier, (n,), keywords=keywords, test_size=.2, X=X, y=y)

    def knn_cross_validate(self, X=None, y=None, normalized=False, keywords={"n_neighbors":5, "weights":"distance",'p':1}, test_size=.3, random_state=5, just_score=False):
      if X == None or y == None:
        X = self.X_no_id
        y = self.y
      if normalized:
        X = preprocessing.normalize(X)
      return self.cross_validate(KNeighborsClassifier, keywords=keywords, X=X, y=y, test_size=test_size, just_score=just_score, random_state=random_state)

    def extra_trees_cross_validate(self, n=50, X=None, y=None, keywords={'random_state':5}, test_size=.3, random_state=5, just_score=False):
      if X == None or y == None:
        X = self.X_no_id
        y = self.y
      #random_state in the non-keywords argument is for the cross_validation data.
      #keyword random_state is for the classifier
      return self.cross_validate(ExtraTreesClassifier, (n,), keywords=keywords, test_size=.1, X=X, y=y, random_state=random_state, just_score=just_score)

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
        x, y = self.separate_target(train_part[i])
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

    def test_data(self, classifier_type, args=(), keywords={}, X=None, y=None, test=None, output_file="submission.csv", probabilities=False, both=False):
      if X==None or y == None:
        X, y = load_training_data()
        X = [x[1:] for x in X] #remove id
      if test == None:
        test = load_test_data()
      indices = [x[0] for x in test]
      test = [x[1:] for x in test]
      
      classifier = classifier_type(*args, **keywords)
      classifier.fit(X, y)
      if probabilities or both:
        predicted = classifier.predict_proba(test)
        max_p = [max(enumerate(x), key = operator.itemgetter(1)) for x in predicted]
        output = [np.concatenate(([x[0]], [x[1][0]+1, x[1][1]], x[2]), 0) for x in zip(indices, max_p, predicted)]
        p_outfile = "proba_{}".format(output_file)
        savetxt(p_outfile, output, delimiter=',', fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f', 
              header='Id,CT,p,Ct1,Ct2,Ct3,Ct4,Ct5,Ct6,Ct7', comments = '')
      if both or not probabilities:
        predicted = classifier.predict(test) 
        output = np.array(zip(indices, predicted), dtype=[('id', int), ('ct',int)])
        savetxt(output_file, output, delimiter=',', fmt='%d,%d', 
              header='Id,Cover_Type', comments = '')
      return True

    def knn_test_data(n=2, X=None, y=None, test=None, output_file="submission.csv", probabilities=False, both=False):
      return test_data(KNeighborsClassifier, (n,), keywords={'p':1, 'weights':'distance'}, X=X, y=y, test=test, output_file=output_file, probabilities=probabilities, both=both)

    def rf_test_data(n=100, X=None, y=None, test=None, output_file="submission.csv", probabilities=False, both=False):
      #Adjusted from https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience tutorial
      return test_data(RandomForestClassifier, keywords={'n_estimators':n, 'random_state':5}, X=X, y=y, test=test, output_file=output_file, probabilities=probabilities, both=both)

    def extra_trees_test_data(self, n=100, X=None, y=None, test=None, output_file="submission.csv", probabilities=False, both=False):
      return self.test_data(ExtraTreesClassifier, keywords={'n_estimators':n, 'max_features':11, 'random_state':5}, X=X, y=y, test=test, output_file=output_file, probabilities=probabilities, both=both)

    def fast_ica(self, X=None, y=None):
      if X == None or y == None:
        X = self.X
        y = self.y
      ica = decomposition.FastICA()
      X_ica = ica.fit_transform(X, y)
      return X_ica
    
    def kernel_pca(self, X=None, y=None):
      if X == None or y == None:
        X = self.X
        y = self.y
      kpca = decomposition.KernelPCA(2, kernel="rbf", gamma=1)
      X_kpca = kpca.fit_transform(X, y)
      return X_kpca

    
    def pca_explore(self, n_components=5, test=False):
      #adapted from http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
      #returns X, y if test=False
      #  if test=True returns X, y, T (transformed training data, target, transformed test data)
      np.random.seed(5)

      pca = decomposition.PCA(n_components)
      pca.fit(self.X_no_id)
      X_new = pca.transform(self.X_no_id)

      if test:
        #in this case returned y not of use
        indices = np.reshape(np.asarray([x[0] for x in self.test]), (len(self.test), 1))
        T = pca.transform([x[1:] for x in self.test])
        T = np.append(indices, T, 1)
        return X_new, self.y, T
      
      return X_new, self.y
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
      
    def knn(keywords={'n_neighbors': 2}):
      out_file="kneighbors_nc_n{}.csv".format(keywords['n_neighbors'])
      X, y = load_training_data(False)
      test = load_test_data(False)
      X = [x[1:] for x in X]
      knn_test_data(output_file = out_file, X=X, y=y, test=test)
      result = "Not yet known(see kaggle)"
      process = "kneighbors not condensed {}".format(keywords)
      return result, out_file, process

    def extra_trees(self, n_estimators=100, X=None, y = None, test=None):
      if X == None or y == None:
        X = self.X
        y = self.y
      if test == None:
        test = self.test
      X_new = np.asarray([x[1:] for x in X])


      result = "Not yet known(see kaggle)"
      out_file="extra_trees{}_mf_11_water_distance.csv".format(n_estimators)
      process = "Extra Trees {}, max features 11, with calculated distance to water".format(n_estimators)
      self.extra_trees_test_data(n=n_estimators, X=X_new, y=y, test=test, output_file=out_file)
      return result, out_file, process

  ##########helper function for keeping track of results
    def keep_track(self, func, args):
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




