import numpy as np
from numpy import genfromtxt, savetxt

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import operator
import random
import math

import kaggle


##########Constants/Labels################################################
COVER = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
FIELDS = ["id", "Elevation", "Aspect", "Slope", "Horizontal distance to water", "Vertical distance to water", "Horizontal distance to roadway", "Hillshade 9am", "Hillshade noon", "Hillshade 3pm", "Horizontal distance to fire points", "Wilderness Area", "Soil Type", "Cover Type"]
WILDERNESS = ["Rawah", "Neota", "Comanche Peak", "Cache La Poudra"]


def load_data():
  data = kaggle.load_training_data(True)
  return data

def plot_example():
  fire = (-600, 300)
  water = (100, 400)
  road = (-200, -300)

  points = [(250, 100), (350, -200), (-100, -50), (100, 500), (-500, 100)]
  
  data = []
  for p in points:
    d = []
    for t in [fire, water, road]:
      d.append(int(dist(t, p)))
    data.append(d)
    print d

  fig = plt.figure(1)
  ax = plt.gca()
  size = 1000
  ax.set_ylim((-1.5*size, 1.5*size))
  ax.set_xlim((-1.5*size, 1.5*size))

  x_vals = [p[0] for p in points]
  y_vals = [p[1] for p in points]
  plt.plot(x_vals, y_vals, 'go')
  plt.plot(fire[0], fire[1], 'ro', markersize=10)
  plt.plot(water[0], water[1], 'bo', markersize=10)
  plt.plot(road[0], road[1], 'ko', markersize=10)

  for i, p in enumerate(points):
    ax.add_artist(plt.Circle((p[0], p[1]), data[i][0], color='r', fill=False))
    ax.add_artist(plt.Circle((p[0], p[1]), data[i][1], color='b', fill=False))
    ax.add_artist(plt.Circle((p[0], p[1]), data[i][2], color='k', fill=False))
  
  plot_points(data, 2, 0, 1, 2, 2)
  return data

def plot_points(data, n=3, f_ind=10, w_ind=4, r_ind=6, fig_n=1):
  #assume data in condensed form from kaggle.load_training/test data
  random.seed(5)

  fig = plt.figure(fig_n)
  ax = plt.gca()
  fire = [x[f_ind] for x in data[0:n]]
  water = [x[w_ind] for x in data[0:n]]
  road = [x[r_ind] for x in data[0:n]]
  size = max((max(fire), max(water), max(road)))
  ax.set_ylim((-1.5*size, 1.5*size))
  ax.set_xlim((-1.5*size, 1.5*size))
  
  x_vals = []
  y_vals = []
  points = []

  for i in range(0, n):
    points.append(next_point(fire, water, road)) #TODO 8/25/14 adjusted how this works, need to fix this part if I end up wanting to keep it.  Working on find_points() at the moment
    x_vals.append(points[i][0])
    y_vals.append(points[i][1])
  plt.plot(x_vals, y_vals, 'go')
  
  for i in range(0, n):
    ax.add_artist(plt.Circle((x_vals[i], y_vals[i]), fire[i], color='r', fill=False))
    ax.add_artist(plt.Circle((x_vals[i], y_vals[i]), water[i], color='b', fill=False))
    ax.add_artist(plt.Circle((x_vals[i], y_vals[i]), road[i], color='k', fill=False))

def find_points(data, n=10, f_ind=10, w_ind=4, r_ind=6):
  random.seed(5)
  fire = [x[f_ind] for x in data]
  water = [x[w_ind] for x in data]
  road = [x[r_ind] for x in data]
  test_points = [(0,0)] #treat first data point as the origin
  error = []

  for i in range(0, n):
    test_points.append(next_point(fire, water, road))
    e = find_min_error(data, test_points[i], fire, water, road)
    #print "test point {}, error {:.5f}".format(test_points[-1], e)
    if e >= 0:
      error.append(e)

  min_index = error.index(min(error))
  print "Min error {:.5f}, point: {}".format(min(error), test_points[min_index])
  max_err = max(error)
  colors = [[e/max_err, 0, 0] for e in error]
  print "Max error: {}:".format(max_err)

  plt.plot([0], [0], 'bo', markersize=10)
  x_vals = [p[0] for p in test_points[1:]]
  y_vals = [p[1] for p in test_points[1:]]
  #plt.plot(x_vals, y_vals, 'go')
  plt.scatter(x_vals, y_vals, c=colors)
  plt.plot(test_points[min_index][0], test_points[min_index][1], 'bo', markersize=10)

  #ax = plt.gca()
  #ax.add_artist(plt.Circle((x_vals[i], y_vals[i]), fire[i], color='r', fill=False))
  




def next_point(fire, water, road):
  max_r = []
  min_r = []
  for t in [fire, water, road]:
    max_r.append(abs(t[0] + t[1]))
    min_r.append(abs(t[0] - t[1]))
  #print "Min R: {}, Max R: {}".format(max(min_r), min(max_r))
  r = random.uniform(max(min_r), min(max_r))
  theta = 0 #random.uniform(0, 2*math.pi)  #Given I am placing point relative to one other point, angle shouldn't matter here
  return (r*math.cos(theta), r*math.sin(theta))

def find_min_error(data, test_point, fire, water, road):
  intersections = find_intersections(test_point, fire, water, road)
  error = []
  if intersections[0] == [] or intersections[1] == [] or intersections[2] == []: #circles don't intersect at all.  May need to adjust this for case where circles are close to intersecting but don't quite
    return -1
  for fp in intersections[0]:
    for w in intersections[1]:
      for r in intersections[2]:
        e = []
        for i in range(2, len(data)):
          e.append(find_distance(fire[i], water[i], road[i], fp, w, r))
        e2 =[v**2 for v in e]  
        error.append(math.sqrt(sum(e2)))
  #print "Min error for test point {} : {:.2f}". format(test_point, min(error))
  return min(error)
  #calculate error for each combination of intersection point possibilities

def find_distance(fire_r, water_r, road_r, fp, w, r):
  #return shortest distance of the intersection of two of the circles to the third circle.  Zero would indicate they all intersect at the same location.
  
  #Three possible combinations of two of the three circles:
  fw = circle_intersections(fp, w, fire_r, water_r) 
  fr = circle_intersections(fp, r, fire_r, road_r)
  wr = circle_intersections(w, r, water_r, road_r)

  distances = []
  for p in fw:
    distances.append(distance_point_circle(p, r, road_r))
  for p in fr:
    distances.append(distance_point_circle(p, w, water_r))
  for p in wr:
    distances.append(distance_point_circle(p, fp, fire_r))
  return min(distances)

def find_intersections(test_point, fire, water, road):
  #Physical location is assumed to be (0, 0) for the first data point.
  #finding intersection of the fire/water/road circles for initial data point and randomly placed (in valid territory) test point
  #test point is a trial for the physical placement of the second data point (relative to the first data point)
  #output is [[fire intersections], [water intersections], [road intersections]].  Each type of intersection could have one or two points.
  #intersections are given as list
  fudge_factor = 0
  intersections = []
  for t in [fire, water, road]:
    intersections.append(circle_intersections((0, 0), test_point, t[0] + fudge_factor, t[1] + fudge_factor))
  return intersections

###############Geometery##############################
def dist(p1, p2):
  return math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def circle_intersections(c1, c2, r1, r2):
  #returns 0, 1 or 2 points of intersection
  #from method described at http://stackoverflow.com/questions/3349125/circle-circle-intersection-points
  #might want to add tolerance buffer for rounding errors with circles that might intersect at a single point
  if not do_circles_intersect(c1, c2, r1, r2):
    return []
  elif c1 == c2 and r1 == r2:
    raise ValueError("Same circle given for both inputs to location.circle_intersections! C1: {}, C2: {}, r1: {}, r2: {}".format(c1, c2, r1, r2))
  else:
    d = dist(c1, c2)
    a = (r1**2 - r2**2 + d**2)/(2*d)
    p2 = list((0,0))
    p2[0] = c1[0] + a*(c2[0] - c1[0])/d
    p2[1] = c1[1] + a*(c2[1] - c1[1])/d
    if d == r1 + r2:
      return [p2]
    else:
      h = math.sqrt(r1**2 - a**2)
      p3 = list((0,0))
      p4 = list((0,0))
      p3[0] = p2[0] + h*(c2[1]-c1[1])/d
      p3[1] = p2[1] - h*(c2[0]-c1[0])/d

      p4[0] = p2[0] - h*(c2[1]-c1[1])/d
      p4[1] = p2[1] + h*(c2[0]-c1[0])/d
      return [p3, p4]

def do_circles_intersect(c1, c2, r1, r2):
  #returns True if they intersect, False 
  #might want to add tolerance buffer for rounding errors with circles that might intersect at a single point
  d = dist(c1, c2)
  if abs(r1-r2) <= d and d <= abs(r1+r2):
    return True
  return False

def distance_point_circle(point, center, radius):
  return abs(dist(point, center) - radius)

if __name__=="__main__":
  run()
