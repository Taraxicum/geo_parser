""" Parser for inverse geographical mapping problem.
(See http://www.datanaturally.com/2015/05/inverse-geographic-mapping-introduction.html for a
description of the problem.)
This parser attempts to find good fitting trios of fixed points (one each of fire, water, road)
then attempts to fit the sample points to the good fixed points, keeping the sample points that
fit well.  The fixed point and sample point sets need to then be fit together (since there is no
way to determine true rotation/reflection/translation we just have to make sure the point sets
are consistent to each other.

    Example Usage:

    import test_data as td #To create test data set
    import parser_fpblanket as fp

    test = td.ForestCoverTestData(1000, 2) #1000 sample points, two sets of fixed points
    parser = fp.BlanketParser(test.data)

    #Attempt to find fixed points that fit well
    parser.test_centers(

    #For the good fixed point sets, find what sample points fit them well
    parser.fit_points_to_fp() 

    #Match up the sets of points that were found to fit well
    cohort, fps = parser.match_on_fp(parser.cohorts_to_match, parser.fp_to_match)
    
    #cohort and fps are now pandas DataFrames
    #cohort corresponds to the sample points and has x, y fields
    #fps corresponds to the predicted fixed points and has x, y, type fields
    
    
    #Since this is an example using test data we can plot against the known
    #true values and see how well it worked.  Note that results we have may need to
    #be translated, rotated, and/or reflected to match up properly with the 
    #original test data.  The following will take care of that before plotting.
    parser.accumulated_cohorts = cohort
    parser.accumulated_fps = fps
    parser.compare_results(test.points, test.fixed_points)
"""
import numpy as np
import pandas as pd
import scipy
import scipy.optimize as opt
from geographic_test import GeoParser
from itertools import chain

import sys

import time

class BlanketParser(GeoParser):
    """ GeoParser which first attempts to find good sets of fixed points, then finds what sample points
    fit those fixed points well, then take those fixed points and sample points and join them into one set
    if possible.
    """
    
    def __init__(self, data, radius=500):
        super(BlanketParser, self).__init__(data, radius)
        self.norm_distance_fields()


    def fit_points_to_fp(self):
        """
        The goal here is if there is a set of 3 fixed points that are believed to be good,
        see what sample points can fit those fixed points well.
        Uses the fixed points in self.fixed_points
        Uses the points in self.data
        return: No return value, but updates cohorts_to_match with the points that fit well
        """
        #Cost/Gradient functions very similar to bgfs_cost/bgfs_gradient, but treat fixed points
        #  as constants.
        #examine_results of the resulting points, use the points that fall within an
        #  acceptable threshold.
        #TODO try out matching smaller sets of points at a time since matching full set is rather 
        #  resource intensive.
        
        self.cohorts_to_match = []
        self.fp_to_match = []
        print "FIT POINTS TO FP - STARTING"
        for i in range(len(self.good_points)):
            matched = set(chain.from_iterable([gc.index for gc in self.cohorts_to_match]))
            current_indices = set(self.good_points[i].index)
            if len(matched) == len(self.data):
                print "Breaking: All points fitted on pass {}".format(i)
                break
            elif len(current_indices & matched) == len(current_indices): #current set adds no new points
                print "Skipping {} since no points added".format(i+1)
                continue
            else:
                print "Fitting {} with {} total matched so far".format(i+1, len(matched))
            self.fixed_points = self.good_fp[i]
            threshold = .5
            self.current_cohort = self.data.loc[self.find_normed_cohort(self.good_points[i].index[0])] #TODO also try with best_n == 0
            print "Running fit_points_to_fp with {} points in current cohort".format(len(self.current_cohort))
            sys.stdout.flush()
            p = self.point_bgfs_call()
            self.update_point_index(p)
            results = self.examine_results(self.current_cohort, p, self.fixed_points)
            keep = results.loc[results.total < threshold].Id
            good = p.loc[p.index.isin(keep)]
            print "For the {}th (of {}) good set fit {} points successfully".format(i + 1, 
                    len(self.good_points), len(good))
            self.cohorts_to_match.append(good)
            self.fp_to_match.append(self.good_fp[i])

    def point_bgfs_call(self, x=None):
        """coordinate the bgfs/gradient descent algorithm for points (fixed points constant)
        fixed points should already be defined in self.fixed_points

        x: vector of hypothesized x, y coordinates for the sample points.  Should be in the
            form [x_1, x_2, ..., x_n, y_1, y_2, ..., y_n]
        """
        #if len(self.current_cohort) != len(self.data):
            #self.current_cohort = self.data
        if x is None:
            x = np.random.randn(len(self.current_cohort)*2)
        print "STARTING point_bgfs_call"
        start_time = time.clock()
        out = opt.fmin_bfgs(self.point_cost, x, self.point_gradient, disp=True)
        points = self.points_from_vector(out)
        end_time = time.clock()
        total_time = end_time - start_time
        print "END point_bgfs_call total time: {}".format(total_time)
        return points
    
    def point_cost(self, x):
        """
        A cost function to compare how well hypothesized x, y coordinates for the sample points 
            fit relative to fixed points in self.fixed_points and their true distances to the
            fixed points as given in Horizontal_Distance_To_Hydrology, etc.
            Used for gradient descent algorithm.
        Same basic cost function as in self.bgfs_cost, but considers fixed points as constants
        which only matters for the gradient.
        x: vector of hypothesized x, y coordinates for the sample points.  Should be in the
            form [x_1, x_2, ..., x_n, y_1, y_2, ..., y_n]
        sets current_cohort to be whole data set
        returns non-negative cost value
        """
        fp_vect = self.fp_to_vector(self.fixed_points)
        cost = self.bgfs_cost(np.append(fp_vect, x))
        return cost
    
    def point_gradient(self, x):
        """Gradient of the point_cost function.  Similar to self.bgfs_gradient, but treats fixed points
        as constants, so ends up being simpler.
        self.fixed_points must have fire, water, and road points defined

        x: vector of hypothesized x, y coordinates for the sample points.  Should be in the
            form [x_1, x_2, ..., x_n, y_1, y_2, ..., y_n]
            To start with, x should be for full sample, maybe in future adjust to allow subsets
        :returns: gradient vector in same format as x
        """
        m = len(x)/2
        if m != len(self.current_cohort):
            print "PROBLEM: GeoParser.point_gradient(x), len(x) should be 2*len(self.current_cohort)"
            return False
        working = pd.DataFrame(x[0:m], columns=['x'])
        working['y'] = x[m:]
        self.update_point_index(working)
        working['fire_x_diff'] = working.x - self.fixed_points.loc[self.fixed_points.type=='fire'].x.iloc[0]
        working['fire_y_diff'] = working.y - self.fixed_points.loc[self.fixed_points.type=='fire'].y.iloc[0]
        working['water_x_diff'] = working.x - self.fixed_points.loc[self.fixed_points.type=='water'].x.iloc[0]
        working['water_y_diff'] = working.y - self.fixed_points.loc[self.fixed_points.type=='water'].y.iloc[0]
        working['road_x_diff'] = working.x - self.fixed_points.loc[self.fixed_points.type=='road'].x.iloc[0]
        working['road_y_diff'] = working.y - self.fixed_points.loc[self.fixed_points.type=='road'].y.iloc[0]
        working['hyp_fire_dist'] = np.sqrt(working.fire_x_diff**2 + working.fire_y_diff**2)
        working['hyp_water_dist'] = np.sqrt(working.water_x_diff**2 + working.water_y_diff**2)
        working['hyp_road_dist'] = np.sqrt(working.road_x_diff**2 + working.road_y_diff**2)
        working['main_grad_fire'] = (
            2*(working.hyp_fire_dist - self.current_cohort.Horizontal_Distance_To_Fire_Points)/working.hyp_fire_dist)
        working['main_grad_water'] = (
            2*(working.hyp_water_dist - self.current_cohort.Horizontal_Distance_To_Hydrology)/working.hyp_water_dist)
        working['main_grad_road'] = (
            2*(working.hyp_road_dist - self.current_cohort.Horizontal_Distance_To_Roadways)/working.hyp_road_dist)
        grad = np.zeros(2*m)
        grad[0:m] = (working['main_grad_fire']*working['fire_x_diff'] + 
            working['main_grad_water']*working['water_x_diff'] + 
            working['main_grad_road']*working['road_x_diff'])/(2*m)
        grad[m:] = (working['main_grad_fire']*working['fire_y_diff'] + 
            working['main_grad_water']*working['water_y_diff'] + 
            working['main_grad_road']*working['road_y_diff'])/(2*m)
        return grad
    
    def get_working_ids(self):
        """Returns set of ids of points that are not already in self.good_points
        """
        if len(self.good_points) == 0:
            return set(self.data.index)
        else:
            return set(self.data.index) - set(chain.from_iterable([p.index for p in self.good_points]))

    def center_iterator(self):
        """iterator over point ids that are not in self.good_points.  It is
        expected that self.good_points may be updated throughout the iteration
        process so updates the point set under consideration each step in the iteration.
        """
        max_checked = -1
        working_ids = self.get_working_ids()
        while len(working_ids) > 0 and max_checked < max(working_ids):
            center_id = min([i for i in working_ids if i > max_checked])
            max_checked = center_id
            yield center_id
            working_ids = self.get_working_ids()
    
    def test_centers(self):
        for c in self.center_iterator():
            self.fixed_points_for(c)
    
    def fixed_points_for(self, center=None, test=None):
        """Attempt to find fixed points for center

        :center: should be index of a point in self.data
        :returns: finds cohort about center, fits, records cost of fit in self.tested_centers
        if cost is less than threshold, records fixed points
        """
        if center is None:
            center = self.data.index[np.random.randint(len(self.data))]
        print "Attempting fixed points for {}".format(center)
        self.current_cohort = self.data.loc[self.find_normed_cohort(center, 6, 6, test)]
        self.fit_current_cohort(center)

