import numpy as np
import pandas as pd
from geographic_test import GeoParser

import time

#################################
# For a general GeoParser class need to implement initialization,
# parsing function, probably some standardized statistics for 
# indication of efficiency.
#
# Base class implements cohort matching, basic cost function
#################################

class BlanketParser(GeoParser):
    """ GeoParser which first attempts to find good sets of fixed points, then finds what sample points
    fit those fixed points well, then take those fixed points and sample points and join them into one set
    if possible.
    """
    
    def fit_points_to_fp(self):
        """
        The goal here is if there is a set of 3 fixed points that are believed to be good,
        see what sample points can fit those fixed points well.
        Uses the fixed points in self.fixed_points
        Uses the points in self.data
        return: No return value, but updates accumulated_cohorts with the points that fit well
        """
        #Cost/Gradient functions very similar to bgfs_cost/bgfs_gradient, but treat fixed points
        #  as constants.
        #examine_results of the resulting points, use the points that fall within an
        #  acceptable threshold.
        #TODO try out matching fewer sets of points at a time since matching full set is rather 
        #  resource intensive.
        
        self.good_cohorts = []
        print "FIT POINTS TO FP - STARTING"
        for i in range(len(self.good_points)):
            self.fixed_points = self.good_fp[i]
            threshold = .5
            p = self.point_bgfs_call()
            results = self.examine_results(self.data, p, self.fixed_points)
            keep = results.loc[results.total < threshold].Id
            good = p.loc[p.index.isin(keep)]
            print "For the {}th (of {}) good set fit {} points successfully".format(i, 
                    len(self.good_points), len(good))
            self.good_cohorts.append(good)

    def point_bgfs_call(self, x=None):
        """coordinate the bgfs/gradient descent algorithm for points (fixed points constant)
        fixed points should already be defined in self.fixed_points

        x: vector of hypothesized x, y coordinates for the sample points.  Should be in the
            form [x_1, x_2, ..., x_n, y_1, y_2, ..., y_n]
        """
        if len(self.current_cohort) != len(self.data):
            self.current_cohort = self.data
        if x is None:
            x = np.random.randn(len(self.data)*2)
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
        return self.bgfs_cost(np.append(fp_vect, x))
    
    def gradient_parts(self, x):
        """Does much of the work of the point_gradient and fp_gradient functions which then sum
        across the appropriate axis of the dataframe returned from this function
        
        x: vector of format [fire_x, water_x, road_x, fire_y, ..., x_1, x_2, ..., x_m, y_1, ..., y_m]
        :returns DataFrame with m rows and includes columns main_fire_x, main_fire_y, ...
        can sum across rows or columns to get gradient for fp or points
        """
        fp = self.fp_from_vector(x[0:6])
        working = self.points_from_vector(x[6:])
        working['fire_x_diff'] = working.x - fp.loc[fp.type=='fire'].x.iloc[0]
        working['fire_y_diff'] = working.y - fp.loc[fp.type=='fire'].y.iloc[0]
        working['water_x_diff'] = working.x - fp.loc[fp.type=='water'].x.iloc[0]
        working['water_y_diff'] = working.y - fp.loc[fp.type=='water'].y.iloc[0]
        working['road_x_diff'] = working.x - fp.loc[fp.type=='road'].x.iloc[0]
        working['road_y_diff'] = working.y - fp.loc[fp.type=='road'].y.iloc[0]
        working['hyp_fire_dist'] = np.sqrt(working.fire_x_diff**2 + working.fire_y_diff**2)
        working['hyp_water_dist'] = np.sqrt(working.water_x_diff**2 + working.water_y_diff**2)
        working['hyp_road_dist'] = np.sqrt(working.road_x_diff**2 + working.road_y_diff**2)
        working['main_fire'] = (
            2*(working.hyp_fire_dist - self.current_cohort.Horizontal_Distance_To_Fire_Points)/working.hyp_fire_dist)
        working['main_water'] = (
            2*(working.hyp_water_dist - self.current_cohort.Horizontal_Distance_To_Hydrology)/working.hyp_water_dist)
        working['main_road'] = (
            2*(working.hyp_road_dist - self.current_cohort.Horizontal_Distance_To_Roadways)/working.hyp_road_dist)
        working['main_fire_x'] = working['main_fire']*working['fire_x_diff']
        working['main_fire_y'] = working['main_fire']*working['fire_y_diff']
        working['main_water_x'] = working['main_water']*working['water_x_diff']
        working['main_water_y'] = working['main_water']*working['water_y_diff']
        working['main_road_x'] = working['main_road']*working['road_x_diff']
        working['main_road_y'] = working['main_road']*working['road_y_diff']
        return working     
    
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
        if m != len(self.data):
            print "PROBLEM: GeoParser.point_gradient(x), len(x) should be 2*len(self.data)"
            return False
        #TODO adjust so that it uses self.gradient_parts and self.current_cohort instead of self.data
        
        working = pd.DataFrame(x[0:m], columns=['x'])
        working['y'] = x[m:]
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
            2*(working.hyp_fire_dist - self.data.Horizontal_Distance_To_Fire_Points)/working.hyp_fire_dist)
        working['main_grad_water'] = (
            2*(working.hyp_water_dist - self.data.Horizontal_Distance_To_Hydrology)/working.hyp_water_dist)
        working['main_grad_road'] = (
            2*(working.hyp_road_dist - self.data.Horizontal_Distance_To_Roadways)/working.hyp_road_dist)
        grad = np.zeros(2*m)
        grad[0:m] = (working['main_grad_fire']*working['fire_x_diff'] + 
            working['main_grad_water']*working['water_x_diff'] + 
            working['main_grad_road']*working['road_x_diff'])/(2*m)
        grad[m:] = (working['main_grad_fire']*working['fire_y_diff'] + 
            working['main_grad_water']*working['water_y_diff'] + 
            working['main_grad_road']*working['road_y_diff'])/(2*m)
        return grad

    def test_all_centers(self):
        for c in self.data.index:
            self.fixed_points_for(c)
    
    def fixed_points_for(self, center=None, test=None):
        """Attmept to find fixed points for center

        :center: should be index of a point in self.data
        :returns: finds cohort about center, fits, records cost of fit in self.tested_centers
        if cost is less than threshold, records fixed points
        """
        if center is None:
            center = self.data.index[np.random.randint(len(self.data))]
        print "Attempting fixed points for {}".format(center)
        self.current_cohort = self.data.loc[self.find_normed_cohort(center, 6, test)]
        self.fit_current_cohort(center)

    def fit_current_cohort(self, center=None):
        """Attempts to fit self.current_cohort and fixed points to x, y coordinates that
        match with the corresponding distance fields (e.g. horizontal_distance_to_roadways)
        In order to be considered successful, cost of fit must be less that self.cost_threshold
        """
        if len(self.current_cohort) < self.cohort_threshold:
            print "FAILED to fit cohort, size got too small with reductions"
            return False
        p, fp = self.init_points(self.current_cohort)
        start_time = time.clock()
        p, fp, cost = self.bgfs_automate()
        end_time = time.clock()
        total_time = end_time - start_time
        if center is not None:
            self.tested_centers[center] = {"cost": cost, "cohort": self.current_cohort.index}
        if cost <= self.cost_threshold:
            print "SUCCESS at fitting cohort of length {} cost {}, time {}".format(len(self.current_cohort), cost, total_time)
            self.good_count += 1
            self.points.append(p)
            self.good_points.append(p)
            self.fixed_points = fp
            self.good_fp.append(fp)
            self.current_cohort['x'] = p.x.values
            self.current_cohort['y'] = p.y.values
            #return self.examine_results(self.current_cohort, p, fp)
            return True
        else:
            print "FAILED to find good fit for cohort of length {} cost {}, time {}".format(len(self.current_cohort), cost, total_time)
            results = self.examine_results(self.current_cohort, p, fp)
            self.remove_problem_points(results)
            try:
                self.accumulated_cohorts.Id
            except AttributeError:
                number_remaining = len(self.current_cohort)
            else:
                number_remaining = len(set(self.current_cohort.Id) - 
                        set(self.accumulated_cohorts.Id))
            if number_remaining > 0:
                print "Retrying: still {} new points in current cohort".format(number_remaining)
                return self.fit_current_cohort(center)
            else:
                print "No more new points in current cohort, abandoning attempt"
                return False
