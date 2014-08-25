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
    points.append(next_point(points, fire, water, road))
    x_vals.append(points[i][0])
    y_vals.append(points[i][1])
  plt.plot(x_vals, y_vals, 'go')
  
  for i in range(0, n):
    ax.add_artist(plt.Circle((x_vals[i], y_vals[i]), fire[i], color='r', fill=False))
    ax.add_artist(plt.Circle((x_vals[i], y_vals[i]), water[i], color='b', fill=False))
    ax.add_artist(plt.Circle((x_vals[i], y_vals[i]), road[i], color='k', fill=False))

def next_point(points, fire, water, road):
  if len(points) == 1:
    max_r = []
    min_r = []
    for t in [fire, water, road]:
      max_r.append(abs(t[0] + t[1]))
      min_r.append(abs(t[0] - t[1]))
    print "Min R: {}, Max R: {}".format(max(min_r), min(max_r))
    r = random.uniform(max(min_r), min(max_r))
    theta = random.uniform(0, 2*math.pi)
    return (points[0][0] + r*math.cos(theta), points[0][1] + r*math.sin(theta))
  else:
    return (-5, -5)

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

if __name__=="__main__":
  run()
