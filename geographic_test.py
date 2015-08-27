""" Parent class of parsers for the inverse geographical mapping problem.
(See http://www.datanaturally.com/2015/05/inverse-geographic-mapping-introduction.html for a
description of the problem.)

See for example parser_fpblanket.py for an implementation using this class.
"""

import numpy as np
import scipy
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import time

#################################
# For a general GeoParser class need to implement initialization,
# parsing function, probably some standardized statistics for 
# indication of efficiency.
#################################

class GeoParser(object):
    """ The tools to try and determine physical x, y coordinates of samples and fixed points of 
        forest cover data.  To date the fields used for input are the horizontal distances to
        fire points, water, and road
        TODO update process description as process stabilizes
    """
    def __init__(self, data, radius=1500):
        self.data = data
        self.radius = radius
        self.norm_radius = 1 #TODO Testing this value, may want to adjust
        self.cohort_threshold = 4  #cohorts smaller than this don't seem to consistently converge correctly
        self.n = len(data)
        self.center_ids = []
        self.accumulated_cohorts = None
        self.accumulated_fps = None #TODO incorporate into rest of code
        self.fixed_points = None
        self.points = []
        self.cost_threshold = 1
        self.tested_centers = {}
        self.good_count = 0
        self.good_points = []
        self.good_cohorts = []
        self.good_fp = []

    
    @staticmethod 
    def init_points(cohort):
        """Initializes set x, y coordinates for hypothesized points and fixed points for iterating 
        :cohort:  DataFrame containing the true distances of the points to the
        :returns: points, fixed_points.  DataFrames of randomized x, y coordinates for each.
        For points it will be the same length as cohort, for fixed points will have x, y
        coordinates and type columns, one row each for fire, water, road.
        """
        points = pd.DataFrame(100*np.random.randn(len(cohort), 2), columns=['x', 'y'])
        vals = 1000*np.random.randn(6)
        fixed_points = pd.DataFrame(vals.reshape((3,2)), columns=['x', 'y'])
        fixed_points['type'] = ['fire', 'water', 'road']
        return points, fixed_points
    
    def fit_current_cohort(self, center=None):
        """Attempts to fit self.current_cohort and fixed points to x, y coordinates that
        match with the corresponding distance fields (e.g. horizontal_distance_to_roadways)
        In order to be considered successful, cost of fit must be less that self.cost_threshold.
        If it succeeds, appends the point set to self.good_cohorts and the fp set to self.good_fp

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
            if number_remaining > self.cohort_threshold:
                print "Retrying: still {} new points in current cohort".format(number_remaining)
                return self.fit_current_cohort(center)
            else:
                print "No more new points in current cohort, abandoning attempt"
                return False


    def remove_problem_points(self, results, percentile=.75):
        """
        :results: DataFrame from self.examine_results contains error sizes for hypothesized coordinates for self.current_cohort.
        :percentile: percentile cutoff to remove points that had larger error

        Doesn't return anything, but sets self.current_cohort to the set of points with error less than the error at percentile.
        The intention here is if the current_cohort did not have a good fit this will cut out the most problematic points so
        the algorithm can be re-run in hopes the fit will be better without the worst fitting points.
        """
        cutoff = results.total.quantile(percentile)
        keep_indices = np.append(results.loc[results.total < cutoff].Id, self.data.loc[self.data.Id.isin(self.center_ids)].Id)
        self.current_cohort = self.current_cohort.loc[self.current_cohort.Id.isin(keep_indices)]

    def norm_distance_fields(self):
        """
        Adds normed_fire, normed_water, normed_road fields to self.data.  These are the normalized values of 
        the Horizontal_Distance_To_Fire_Points, Horizontal_Distance_To_Hydrology, Horizontal_Distance_To_Roadways fields
        in self.data.
        The intention here is that the normed fields might be useful for finding cohorts about a center since the range of 
        values is different for each of the horizontal distance fields.  In looking a at a few examples it didn't look like
        it made a lot of difference in what points ended up in a cohort, though the ordering did change a little for some points.
        """
        fp = self.data.Horizontal_Distance_To_Fire_Points
        self.data['normed_fire'] = (fp - fp.mean())/fp.std()
        fp = self.data.Horizontal_Distance_To_Hydrology
        self.data['normed_water'] = (fp - fp.mean())/fp.std()
        fp = self.data.Horizontal_Distance_To_Roadways
        self.data['normed_road'] = (fp - fp.mean())/fp.std()

    def find_normed_cohort(self, pid, best_n = 0, max_n = 100, test=None):
        """ Finds cohort using normed distance fields.
            Similar to find_cohort but using normalized fields
            pid: value of Id field of center point of cohort
            best_n: if best_n == 0 return all points within self.norm_radius,
                if best_n > 0 then return the top best_n fitting points
            If test is defined (should be the ForestCoverTestData object that was used to generate self.data) prints out
            the best fitting best_n points (prints 20 if best_n <= 0) with which reference points each was closest to in
            the original test data.  This can show if the cohort is likely to have a good fit because for instance if some
            of the points in the cohort were originally closest to fire_point 1 and others were closest to fire_point 2
            they may well have not actually been close in the original set.
        """
        water_v_threshold = 5
        fds = pd.DataFrame(columns=['distance'])
        locus = self.data.loc[self.data.Id == pid]
        #First filter by if water source could be same - i.e. Vertical_Distance_To_Hydrology - Elevation is same (close to same?)
        water_elev = (locus.Vertical_Distance_To_Hydrology - locus.Elevation).values[0]
        filter_indices = self.data.loc[
                abs((self.data.Vertical_Distance_To_Hydrology - self.data.Elevation) - water_elev) < water_v_threshold].index
        
        fdata = self.data.loc[filter_indices]
        fds.distance = (locus.normed_fire.values[0] - fdata.normed_fire)**2
        fds.distance += (locus.normed_road.values[0] - fdata.normed_road)**2
        fds.distance += (locus.normed_water.values[0] -  fdata.normed_water)**2
        fds.distance = np.sqrt(fds.distance)
        ########TEMP####################
        if test is not None:
            fds['fire_index'] = test.points.fire_index
            fds['water_index'] = test.points.water_index
            fds['road_index'] = test.points.road_index
            if best_n > 0:
                print fds.sort('distance').head(best_n)
            else:
                print fds.loc[fds.distance < self.norm_radius].sort('distance').head(20)
        #print fds.loc[(fds.fire_index == fds.loc[pid].fire_index) & (fds.water_index == fds.loc[pid].water_index)
        #        & (fds.road_index == fds.loc[pid].road_index)].sort('distance').head(20)
        ########TEMP####################
        if best_n == 0:
            return fds.loc[fds.distance < self.norm_radius].head(max_n).index
        elif best_n > 0:
            return fds.sort('distance').head(best_n).index
        else:
            print "GeoParser.find_normed_cohort: Invalid value for best_n should be non negative integer, got {}".format(best_n)
            return
            
    
    def find_cohort(self, pid, test=None): 
        """ Find cohort of points that are potentially close to the center using elevation, vertical distance to hydrology
        as an initial filter, then using the differences in horizontal distance fields as proxies for closeness.
        Similar to find_normed_cohort but using un-normalized fields
            :pid: value of Id field of center point of cohort
        """
        water_v_threshold = 5
        #ds = pd.DataFrame(np.zeros((self.n, 1)), columns=['distance'])
        fds = pd.DataFrame(columns=['distance'])
        locus = self.data.loc[self.data.Id == pid]
        #First filter by if water source could be same - i.e. Vertical_Distance_To_Hydrology - Elevation is same (close to same?)
        water_elev = (locus.Vertical_Distance_To_Hydrology - locus.Elevation).values[0]
        filter_indices = self.data.loc[
                (self.data.Vertical_Distance_To_Hydrology - self.data.Elevation) - water_elev < water_v_threshold].index
        #fds = ds.loc[filter_indices]
        fdata = self.data.loc[filter_indices]
        fds.distance = (locus.Horizontal_Distance_To_Fire_Points.values[0] - fdata.Horizontal_Distance_To_Fire_Points)**2
        fds.distance += (locus.Horizontal_Distance_To_Roadways.values[0] - fdata.Horizontal_Distance_To_Roadways)**2
        fds.distance += (locus.Horizontal_Distance_To_Hydrology.values[0] -  fdata.Horizontal_Distance_To_Hydrology)**2
        fds.distance = np.sqrt(fds.distance)
        #########TEMP####################
        #fds['fire_index'] = test.points.fire_index
        #fds['water_index'] = test.points.water_index
        #fds['road_index'] = test.points.road_index
        #print fds.sort('distance').head(20)
        #########TEMP####################
        return fds.loc[fds.distance < self.radius].index

    def bgfs_automate(self, p=None, fp=None):
        if (p is None) or (fp is None):
            p, fp = self.bgfs_call()
        else:
            x = self.points_fp_to_vector(p, fp)
            cost = self.cost(self.current_cohort, p, fp)
            p, fp = self.bgfs_call(x)
        self.update_point_index(p)
        cost = self.cost(self.current_cohort, p, fp)
        return p, fp, cost
    
    def update_point_index(self, points):
        """Updates index of points inplace with the index of self.current_cohort.
        Note that points and self.current_cohort should be of same length since the points
        should have come from fitting self.current_cohort to x, y values.
        """
        points['data_index'] = self.current_cohort.index
        points.set_index('data_index', inplace=True)
        points.index.name = None

    def bgfs_call(self, x = None):
        if x is None:
            x = np.random.randn(len(self.current_cohort)*2 + 6)
        out = opt.fmin_bfgs(self.bgfs_cost, x, self.bgfs_gradient, disp=False)
        fp = self.fp_from_vector(out[0:6])
        points = self.points_from_vector(out[6:])
        return points, fp
    
    def bgfs_cost(self, x):
        #x should have first 6 arguments be fire x, y, water x, y, road x,y, then hypothesized x-vals then y-vals
        fixed_points = self.fp_from_vector(x[0:6])
        points = self.points_from_vector(x[6:])
        #TODO Test for other methods than just fpblanket
        self.update_point_index(points)
        fire_d = (self.dist(points, fixed_points[fixed_points.type == 'fire']) -
                self.current_cohort.Horizontal_Distance_To_Fire_Points.values)**2
        water_d = (self.dist(points, fixed_points[fixed_points.type == 'water']) -
                self.current_cohort.Horizontal_Distance_To_Hydrology.values)**2
        road_d = (self.dist(points, fixed_points[fixed_points.type == 'road']) -
                self.current_cohort.Horizontal_Distance_To_Roadways.values)**2
        return 1.0/(2*len(self.current_cohort))*(fire_d.sum() + water_d.sum() + road_d.sum())
    
    def bgfs_gradient(self, x):
        #x should have first 6 arguments be fire x, y, water x, y, road x,y, then hypothesized x-vals then y-vals
        fp = self.fp_from_vector(x[0:6])
        points = self.points_from_vector(x[6:])
        px, py, fixed = self.cost_deriv(self.current_cohort, points, fp)
        fv = self.fp_to_vector(fixed)
        return np.concatenate([fv, px.values, py.values])
    
    def cost(self, cohort, points, fixed_points):
        """Returns non-negative real number
            cohort of points we are trying to map to a 2d representation that fits the data
            points are the x, y coordinates of the hypothesized points (should be len(cohort) of them)
            fixed_points x, y coordinates of the hypothesized fire, water, road locations
            This is not the same cost function the derivative of the cost function uses - this 
            one uses square root to make the values a little easier to think of as an average error
            but would unnecessarily complicate the derivative
        """
        fire_d = np.sqrt((self.dist(points, fixed_points[fixed_points.type == 'fire']) -
            self.current_cohort.Horizontal_Distance_To_Fire_Points.values)**2)
        water_d = np.sqrt((self.dist(points, fixed_points[fixed_points.type == 'water']) -
            self.current_cohort.Horizontal_Distance_To_Hydrology.values)**2)
        road_d = np.sqrt((self.dist(points, fixed_points[fixed_points.type == 'road']) -
            self.current_cohort.Horizontal_Distance_To_Roadways.values)**2)
        return 1.0/(2*len(cohort))*(fire_d.sum() + water_d.sum() + road_d.sum())
    
    @classmethod
    def distance_xy(cls, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


    def dist(self, points, fp):
        #TODO classmethod?
        #Distance between points and fixed point (fire, water, road).
        #points: should be a DataFrame 
        #fp: should be a single fixed point object (e.g. has x, y, type properties)
        fpi = fp.iloc[0].name
        return np.sqrt((points.x - fp.loc[fpi, 'x'])**2 + (points.y - fp.loc[fpi, 'y'])**2)

    def partial(self, cohort_d, points, fp):
        #TODO classmethod?
        #Finds the partial derivative of the part of the cost function relating to the fixed point fp
        #cohort_d: Series containing the true distances of the points to the true fixed point
        #points:  DataFrame of the hypothesized points
        #fp:  dict of a single of the hypothesized fixed points e.g. {fire: {'x':1, 'y':2}}
        distances = self.dist(points, fp) 
        differences = distances - cohort_d.values
        main_partial = (differences/distances)
        partial_x = main_partial*2*(points.x - fp.iloc[0]['x'])
        partial_y = main_partial*2*(points.y - fp.iloc[0]['y'])
        return partial_x, partial_y
        
    def cost_deriv(self, cohort, points, fixed_points):
        #TODO classmethod?
        #Returns the partial derivatives of the cost function relative to the hypothesized x, y and fixed points
        #cohort:  DataFrame of the true distances
        #points:  DataFrame of the hypothesized x,y coordinates
        #fixed_points:  dict of the fixed points (fire, water, road)
        fixed = pd.DataFrame(np.zeros((3,3)), columns=['x', 'y', 'type'])
        f_p_x, f_p_y = self.partial(cohort.Horizontal_Distance_To_Fire_Points, points, fixed_points[fixed_points.type == 'fire'])
        w_p_x, w_p_y = self.partial(cohort.Horizontal_Distance_To_Hydrology, points, fixed_points[fixed_points.type == 'water'])
        r_p_x, r_p_y = self.partial(cohort.Horizontal_Distance_To_Roadways, points, fixed_points[fixed_points.type == 'road'])
        a = 1.0/(2*len(cohort))
        fixed.loc[0, ['type', 'x', 'y']] = ['fire', -a*f_p_x.sum(), -a*f_p_y.sum()]
        fixed.loc[1, ['type', 'x', 'y']] = ['water', -a*w_p_x.sum(), -a*w_p_y.sum()]
        fixed.loc[2, ['type', 'x', 'y']] = ['road', -a*r_p_x.sum(), -a*r_p_y.sum()]
        partial_x = a*(f_p_x + w_p_x + r_p_x)
        partial_y = a*(f_p_y + w_p_y + r_p_y)
        return partial_x, partial_y, fixed

    def recenter(self, fixed_points, points, x_amount, y_amount):
        #TODO classmethod?
        #The reason for this function is to make comparison of original points and hypothesized points simpler
        #  because we can center around a corresponding point (e.g. set fire point to 0,0 for both sets)
        #  and then matching up their plots only requires reflection and/or rotation
        #This may also be useful (needs more testing) for helping get hypothesized points out of local minima
        #  by centering on a fixed point and reflecting problem point across it
        if points is not None:
            points.x += x_amount
            points.y += y_amount
        if fixed_points is not None:
            fixed_points.x += x_amount
            fixed_points.y += y_amount
        
    @classmethod
    def points_fp_to_vector(cls, points, fp):
        return np.concatenate([cls.fp_to_vector(fp), cls.points_to_vector(points)])

    @classmethod
    def points_to_vector(cls, points):
        return np.concatenate([points.x.values, points.y.values])

    @classmethod
    def points_from_vector(cls, x):
        """
        x: list of length 2m consisting of x, y coordinates with the x-coordinates
            in the first m places, the y-coordinates in the second m places
        returns DataFrame of m points with appropriate x, y columns
        """
        m = len(x)/2
        points = pd.DataFrame(np.zeros((m, 2)), columns=['x', 'y'])
        points.x = x[0:m]
        points.y = x[m:]
        return points


    @staticmethod
    def fp_to_vector(fp):
        x = np.zeros(6)
        x[0] = fp.loc[fp.type == 'fire'].iloc[0]['x']
        x[1] = fp.loc[fp.type == 'fire'].iloc[0]['y']
        x[2] = fp.loc[fp.type == 'water'].iloc[0]['x']
        x[3] = fp.loc[fp.type == 'water'].iloc[0]['y']
        x[4] = fp.loc[fp.type == 'road'].iloc[0]['x']
        x[5] = fp.loc[fp.type == 'road'].iloc[0]['y']
        return x

    @staticmethod
    def fp_from_vector(x):
        """
        fp: list of length 6 consisting of alternating x, y coordinates in the order:
            fire, water, road
        returns DataFrame of m points with appropriate x, y, type columns
        """
        fp = pd.DataFrame(np.zeros((3,3,)), columns=['x', 'y', 'type'])
        fp.loc[0, 'type'] = 'fire'
        fp.loc[1, 'type'] = 'water'
        fp.loc[2, 'type'] = 'road'
        fp[['x', 'y']] = np.reshape(x, (3, 2))
        return fp

    def align_cohorts(self, primary, secondary, secondary_fp):
        #primary cohort will have the point values maintained
        #secondary cohort will have point values adjusted to line up with primary if possible
        #secondary fixed points will receive same transformations as secondary cohort
        #There must be at least three points in common between primary and secondary for it to work.
        overlap = primary.index & secondary.index
        if len(overlap) < 3:
            return False
        #First shift secondary so one of the points aligns with corresponding primary point
        diff_x = primary.loc[overlap[0]].x - secondary.loc[overlap[0]].x
        diff_y = primary.loc[overlap[0]].y - secondary.loc[overlap[0]].y
        self.recenter(secondary_fp, secondary, diff_x, diff_y)
        
        #Next need rotation/reflection that best fits remaining overlapped points
        #We want the rotation to be around the point of commonality, so easiest if we
        #  shift that value to 0, 0, do the rotation, then shift back
        shift_x = primary.loc[overlap[0]].x
        shift_y = primary.loc[overlap[0]].y
        self.recenter(None, primary, -shift_x, -shift_y)
        self.recenter(secondary_fp, secondary, -shift_x, -shift_y)
        
        #Normal equation: Ax = y ->  Ax'x = x'y ->  A = (x'x)^-1 x'y
        #FIXME I believe there is a problem in the rotation part of this alignment algorithm
        #  I think the problem is does not find rigid body transformation.  Probably better
        #  to find angle between vectors, then use it for the rotation
        #TODO use find_rotation to get rigid body transformation matrix
        X = secondary.loc[overlap, ['x', 'y']].values
        Y = primary.loc[overlap, ['x', 'y']].values
        transform = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), Y))
        secondary[['x', 'y']] = np.dot(secondary[['x', 'y']], transform)
        #Need to rotate fixed points as well
        secondary_fp[['x', 'y']] = np.dot(secondary_fp[['x', 'y']], transform)
        self.recenter(None, primary, shift_x, shift_y)
        self.recenter(secondary_fp, secondary, shift_x, shift_y)
        return True
    
    @classmethod
    def examine_results(cls, cohort, p, fp):
        """
        cohort is original data set, p is hypothesized points, fp is hypothesized fire, water, road points
        returns the difference between the hypothesized distances and true distances 
          between each point and the fixed points rounded to integer
        """
        if len(cohort) != len(p):
            print("ERROR in GeoParser.examine_results.  Cohort and p should be of same length but got cohort length {}"
                    " and p length {}".format(len(cohort), len(p)))
            return None
  
        ordering = pd.DataFrame(np.zeros((len(p), 4)), columns=['fire', 'water', 'road', 'total'], index=cohort.index)
        ordering['Id'] = ordering.index
        fire = fp[fp.type == 'fire'].iloc[0]
        water = fp[fp.type == 'water'].iloc[0]
        road = fp[fp.type == 'road'].iloc[0]
        ordering['fire'] = np.sqrt((p.x - fire['x'])**2 + (p.y - fire['y'])**2) - cohort.Horizontal_Distance_To_Fire_Points
        ordering['water'] = np.sqrt((p.x - water['x'])**2 + (p.y - water['y'])**2) - cohort.Horizontal_Distance_To_Hydrology
        ordering['road'] = np.sqrt((p.x - road['x'])**2 + (p.y - road['y'])**2) - cohort.Horizontal_Distance_To_Roadways
        ordering['total'] = abs(ordering['fire']) + abs(ordering['water']) + abs(ordering['road'])
        return ordering


    @classmethod
    def remove_duplicate_sets(cls, cohorts, fps):
        working_cohorts = []
        working_fps = []
        point_set = set()
        count = 0
        for i in range(len(cohorts)):
            point_set = point_set.union(set(cohorts[i].index))
            if len(point_set) != count: #added new points
                working_cohorts.append(cohorts[i])
                working_fps.append(fps[i])
                count = len(point_set)
        return working_cohorts, working_fps

    @classmethod
    def match_on_fp(cls, cohorts, fps):
        """
        :cohorts should be a list of working sample point cohorts with good fits
        :fps should be a list of working fixed points with good fits corresponding to the cohorts
        :returns accumulated_cohort, accumulated_fps: single cohort/fixed point set that accumulated
        all points/fixed points determined to be unique (i.e. doesn't double include any points)
        and were able to be aligned so they would be in same rotation/translation/reflection orientation
        """
        if len(cohorts) != len(fps):
            print("Problem in match_on_fp():  cohorts and fps should be lists of same length instead got "
                    "len(cohort) = {}, len(fps) = {}".format(len(cohorts), len(fps)))
            return None
        distance_threshold = 50
        
        working_cohorts, working_fps = cls.remove_duplicate_sets(cohorts, fps)
        accumulated_cohort = pd.DataFrame(working_cohorts[0], columns = working_cohorts[0].columns)
        accumulated_fps = pd.DataFrame(working_fps[0], columns=working_fps[0].columns)
        
        cohort_count = len(accumulated_cohort)
        fp_count = len(accumulated_fps)
        accumulated_cohort, accumulated_fps, remaining_indices = cls.accumulate_cohorts_fp(accumulated_cohort,
                accumulated_fps, working_cohorts[1:], working_fps[1:])
        
        print "fp_count: {}; len(accumulated_fps: {}".format(fp_count, len(accumulated_fps))
        print "cohort_count: {}; len(accumulated_cohort: {}".format(cohort_count, len(accumulated_cohort))
        
        while (len(remaining_indices) > 0 and 
                (fp_count < len(accumulated_fps) or cohort_count < len(accumulated_cohort))):
            #Didn't get all cohorts merged - if there were any new points merged together,
            #  try again as maybe they will match up now.
            cohort_count = len(accumulated_cohort)
            fp_count = len(accumulated_fps)
            working_cohorts = [c for i, c in enumerate(working_cohorts) if i in remaining_indices]
            working_fps = [fp for i, fp in enumerate(working_fps) if i in remaining_indices]
            accumulated_cohort, accumulated_fps, remaining_indices = cls.accumulate_cohorts_fp(accumulated_cohort,
                    accumulated_fps, working_cohorts[1:], working_fps[1:])
            print "fp_count: {}; len(accumulated_fps: {}".format(fp_count, len(accumulated_fps))
            print "cohort_count: {}; len(accumulated_cohort: {}".format(cohort_count, len(accumulated_cohort))
        print "remaining indices after match_on_fp: {}".format(remaining_indices)
        return accumulated_cohort, accumulated_fps
    
    @classmethod
    def accumulate_cohorts_fp(cls, accumulated_cohort, accumulated_fps, working_cohorts, working_fps):
        remaining_indices = []
        for i in range(1,len(working_cohorts)):
            #for j in range(i, len(working_cohorts)):
            #cls.count += 1
            #print "Starting to accumulate step {}".format(cls.count)
            success, remainingfp = cls.align_pair_on_fp(working_cohorts[i], working_fps[i],
                    accumulated_cohort, accumulated_fps)
            if success:
                accumulated_cohort = accumulated_cohort.combine_first(working_cohorts[i])
                accumulated_fps = accumulated_fps.append(working_fps[i].loc[remainingfp], ignore_index=True)
            else:
                remaining_indices.append(i)
                print "Failed to match up working cohort {} with accumulated cohort".format(i)
        return accumulated_cohort, accumulated_fps, remaining_indices 
    
    @classmethod
    def find_overlapped_fp(cls, fp1, fp2, distance_threshold=50):
        """Finds points between cohort1, cohort2 and fp1, fp2 that are likely overlapping points
        
        :fp1, fp2 the fixed points to compare (each should be a dataframe).
        See distance_threshold for a description of how they are matched.

        :distance_threshold to compare fixed points this function compares the distance between fixed
        points in the set.  For example if in fp1 the distance between a fire point and water point is
        3200 and in fp2 there is a fire point, water point pair that is 3180 apart from each other, we
        might want to try treating the fire point and water point from each as the same points.

        :returns: overlap1, overlap2, remaining ->  DataFrames with the x, y values of
        prospective overlapped fixed points.  Should be in the same order so overlap1.iloc[i] corresponds
        to overlap2.iloc[i].  remaining is the list of indices from fp1 that are not included in the 
        overlap.
        """
        best_fits = {}
        to_remove = []
        array1 = []
        array2 = []
        for i in range(len(fp1)):
            for j in range(i + 1, len(fp1)):
                (dist, pair2) = cls.fixed_point_match(fp1.iloc[i], fp1.iloc[j], fp2)
                if 0 <= dist < distance_threshold:
                    best_fits[(fp1.iloc[i].name, fp1.iloc[j].name)] = {'cost': dist, 'match': pair2}
        if best_fits != {}:
            already_added = {}
            for k in best_fits.keys():
                if not k[0] in already_added:
                    already_added[k[0]] = True #It seems like there should be a better way of doing this
                    #since I don't actually use the value in the dict, just the existence of the key.
                    array1.append(fp1.loc[k[0], ['x', 'y']].values)
                    array2.append(fp2.loc[best_fits[k]['match'][0], ['x', 'y']].values)
                    to_remove.append(k[0])
                if not k[1] in already_added:
                    already_added[k[1]] = True
                    array1.append(fp1.loc[k[1], ['x', 'y']].values)
                    array2.append(fp2.loc[best_fits[k]['match'][1], ['x', 'y']].values)
                    to_remove.append(k[1])
            overlap1 = pd.DataFrame(array1, columns=['x', 'y'])
            overlap2 = pd.DataFrame(array2, columns=['x', 'y'])
            remaining_indices1 = np.delete(np.asarray(fp1.index.copy()), to_remove)
        else:
            overlap1 = pd.DataFrame(columns=['x', 'y'])
            overlap2 = pd.DataFrame(columns=['x', 'y'])
            remaining_indices1 = fp1.index
        return overlap1, overlap2, remaining_indices1

    @classmethod
    def find_overlapped_points(cls, cohort1, cohort2):
        """
        :cohort1, cohort2:  two dataframes with x, y columns to be compared for overlapping indices
        :returns overlap1, overlap2: DataFrames that are the subsets of cohort1, cohort2 with just
        the rows that have overlapping indices.  Empty DataFrames with columns 'x', 'y' if no overlap.
        """
        array1 = []
        array2 = []
        matching_indices = set(cohort1.index) & set(cohort2.index)
        if matching_indices != set([]):
            overlap1 = cohort1.loc[matching_indices]
            overlap2 = cohort2.loc[matching_indices]
        else:
            overlap1 = pd.DataFrame(columns=['x', 'y'])
            overlap2 = pd.DataFrame(columns=['x', 'y'])

        return overlap1, overlap2



    @classmethod
    def align_pair_on_fp(cls, cohort1, fp1, cohort2, fp2):
        """
        align points by finding matching points in cohorts and fixed points
            to find best rigid transformations (e.g. rotation, reflection, translation) to align
        :fp1, fp2 fixed points may contain more than 1 of any type but must have at least one of each
            (e.g. could have 3 water points, 1 fire point, 2 road points).  Should be DataFrames
        :cohort1, cohort2 DataFrames of points desired to be aligned
        
        :returns True and fits cohorts/fp if sufficient overlap is found
        :returns False otherwise
        #:returns cohort, fp that combines the cohorts/fp using the best fit of the pairs
        """
        #Try to align on fixed points first
        overlapfp1, overlapfp2, remaining1 = cls.find_overlapped_fp(fp1, fp2)
        overlap1, overlap2 = cls.find_overlapped_points(cohort1, cohort2)
        overlap1 = overlap1.append(overlapfp1)
        overlap2 = overlap2.append(overlapfp2)
        #print "OVERLAP1 {}".format(overlap1)
        #print "OVERLAP2 {}".format(overlap2)
        if overlap1 is not None and len(overlap1.index) >= 2:
            cls.fit_using_overlap(overlap1, overlap2, cohort1, cohort2, fp1, fp2)
            if len(overlap1.index) == 2: #May need to be reflected before fitting
                reflect = ""
                while reflect.lower() != "n":
                    plot_two(cohort1, cohort2, fp1, fp2)
                    plt.show()
                    reflect = raw_input("Perform reflection? y/n/s")
                    if reflect.lower() == "y":
                        cls.fit_using_overlap(overlap1, overlap2, cohort1, cohort2, fp1, fp2, reflect=True)
                    if reflect.lower() == "s":
                        return False, remaining1
            return True, remaining1

        return False, remaining1

    @classmethod
    def fixed_point_match(cls, fp1a, fp1b, fp_set):
        """Trying to match up two of the fixed points (fp1a, fp1b) from one set with
        a second set (fp_set).

        :fp1a, fp1b fixed points from the same set 
        :fp_set a second set of fixed points to attempt to fit to fp1a, fp1b
        :returns: closest, best.  Closest is the distance of the best fit between fp1a, fp1b and a
        pair of fixed points from fp_set.  Best is an ordered pair of the indices of the pair of 
        fixed points from fp_set that fit fp1a and fp1b best.
        """
        dist1 = cls.distance_xy(fp1a, fp1b)
        fp2a = fp_set[fp_set.type == fp1a.type]
        fp2b = fp_set[fp_set.type == fp1b.type]
        closest = -1
        best = None
        for i, a in fp2a.iterrows():
            for j, b in fp2b.iterrows():
                dist2 = cls.distance_xy(a, b)
                val = abs(dist1 - dist2)
                if closest < 0 or val < closest:
                    closest = val
                    best = (a.name, b.name)
        return closest, best

    @classmethod
    def find_rotation(cls, A, B):
        """Find rotation matrix to get best rigid body transformation to fit points in A to B.
        Points in A and B should have already been translated to have centroids at origin.
        Use svd rigid transformation best fit methodology as described at http://nghiaho.com/?page_id=671
        or at https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        
        :A: DataFrame with x, y fields to be rotated to fit points in B.
        :B: DataFrame with x, y fields that A is being fit to.  A and B need to be same length.
        :returns: rot, det: 2x2 rotation matrix such that dot(rotation, A) is approximately B, 
        and its determinant (should be +-1)

        """
        a = A[['x', 'y']].values
        b = B[['x', 'y']].values
        H = np.dot(a.T, b)
        U, S, V = np.linalg.svd(H)
        
        rot = np.dot(U, V.T)
        #Because I want to multiply by the rotation on the right I want it transposed 
        #Appear to not be the case!
        #rot = np.transpose(rot)
        det = np.linalg.det(rot)
        return rot, det
    
    @classmethod
    def fit_using_overlap(cls, overlap1, overlap2, points1, points2, fp1, fp2, reflect=False):
        #shift so overlap centroids are at the origin
        centroid1 = overlap1.mean()
        centroid2 = overlap2.mean()
        overlap1 -= centroid1
        overlap2 -= centroid2
        points1[['x', 'y']] -= centroid1
        points2[['x', 'y']] -= centroid2
        fp1[['x', 'y']] -= centroid1
        fp2[['x', 'y']] -= centroid2
        
        #print overlap1
        #print overlap2
        #if reflect == True:
            #fp2.y = -fp2.y
            #points2.y = -points2.y
            #overlap2.y = -overlap2.y

        rot, det = cls.find_rotation(overlap2, overlap1)
        #print "determinant {}, rotation {}".format(det, rot)
        if det < 0 and len(overlap1) > 2:
            #There needs to be a reflection to get points correctly aligned.  I'm not sure
            #why we can't just apply rot as if its det = -1 it should have a reflection as
            #part of the transformation, but for some reason that doesn't work.
            #Reflecting the points and finding a new rotation does work though, so I will
            #stick with that for now.  I would like to understand what is going on better at some
            #point though.
            fp2.y = -fp2.y
            points2.y = -points2.y
            overlap2.y = -overlap2.y
            rot, det = cls.find_rotation(overlap2, overlap1)
            #print "new determinant {}, rotation {}".format(det, rot)
        #if len(overlap1) == 2: #FIXME Something weird going on sometimes when only 2 overlap, this
        #is an attempt to figure out something that works
            #points2[['x', 'y']] = np.dot(points2[['x', 'y']], -rot)
            #fp2[['x', 'y']] = np.dot(fp2[['x', 'y']], -rot)
        #else:
        overlap2[['x', 'y']] = np.dot(overlap2[['x', 'y']], rot)
        points2[['x', 'y']] = np.dot(points2[['x', 'y']], rot)
        fp2[['x', 'y']] = np.dot(fp2[['x', 'y']], rot)

# Functions to create test data and display results
    def plot_results(self):
        """Plot hypothesized points/fp as found in self.accumulated_cohorts and 
        self.accumulated_fps.
        While getting code setup with accumulated_fps will first check for
        existence before attempting to plot the fps.
        """
        fp_size = 450
        true_p_size = 60
        hyp_p_size = 150

        points = self.accumulated_cohorts
        plt.scatter(points.x, points.y, c='orange', marker='x', s=hyp_p_size)
        
        if self.accumulated_fps is not None:
            fps = self.accumulated_fps
            plt.scatter(fps.loc[fps.type == 'fire'].x.values, 
                    fps.loc[fps.type == 'fire'].y.values, c='red', marker='x', s=fp_size)
            plt.scatter( fps.loc[fps.type == 'water'].x.values, 
                    fps.loc[fps.type == 'water'].y.values, c='blue', marker='x', s=fp_size)
            plt.scatter( fps.loc[fps.type == 'road'].x.values, 
                fps.loc[fps.type == 'road'].y.values, c='black', marker='x',  s=fp_size)

    @classmethod
    def plot_two(cls, points1, points2, fp1, fp2):
        """Plot two sets of points, fixed points.  First set is drawn as 'o',
        second set is drawn as 'x'.
        """
        fp_size = 450
        true_p_size = 60
        hyp_p_size = 150
        plt.scatter(points1.x, points1.y, c='green', marker='o', s=true_p_size)
        plt.scatter(fp1.loc[fp1.type == 'fire'].x.values, 
                fp1.loc[fp1.type == 'fire'].y.values, c='red', s=fp_size)
        plt.scatter(fp1.loc[fp1.type == 'water'].x.values, 
                fp1.loc[fp1.type == 'water'].y.values, c='blue', s=fp_size)
        plt.scatter(fp1.loc[fp1.type == 'road'].x.values, 
                fp1.loc[fp1.type == 'road'].y.values, c='black', s=fp_size)
        
        plt.scatter(points2.x, points2.y, c='orange', marker='x', s=hyp_p_size)
        if fp2 is not None:
            plt.scatter(fp2.loc[fp2.type == 'fire'].x.values, 
                    fp2.loc[fp2.type == 'fire'].y.values, c='red', marker='x', s=fp_size)
            plt.scatter(fp2.loc[fp2.type == 'water'].x.values, 
                    fp2.loc[fp2.type == 'water'].y.values, c='blue', marker='x', s=fp_size)
            plt.scatter(fp2.loc[fp2.type == 'road'].x.values, 
                    fp2.loc[fp2.type == 'road'].y.values, c='black', marker='x',  s=fp_size)

            
    def compare_results(self, true_points, true_fps):
        """
          Plot the hypothesized points and fixed points as well as the true points and fixed points.
          The true values will plot as circles, the hypothesized as x.
          The fixed points will be larger with same shape scheme and red for fire, blue for water, black for road
          Hypothesized points should be found in self.accumulated_cohorts and self.accumulated_fps
        """
        fp_size = 450
        true_p_size = 60
        hyp_p_size = 150
        
        true_points = true_points.copy()
        true_fps = true_fps.copy()
        points = self.accumulated_cohorts.copy()
        if self.accumulated_fps is not None:
            fps = self.accumulated_fps.copy()
        self.align_pair_on_fp(true_points, true_fps, points, fps)

        plt.scatter(true_points.x, true_points.y, c='green', marker='o', s=true_p_size)
        plt.scatter(true_fps.loc[true_fps.type == 'fire'].x.values, 
                true_fps.loc[true_fps.type == 'fire'].y.values, c='red', s=fp_size)
        plt.scatter(true_fps.loc[true_fps.type == 'water'].x.values, 
                true_fps.loc[true_fps.type == 'water'].y.values, c='blue', s=fp_size)
        plt.scatter(true_fps.loc[true_fps.type == 'road'].x.values, 
                true_fps.loc[true_fps.type == 'road'].y.values, c='black', s=fp_size)
        
        plt.scatter(points.x, points.y, c='orange', marker='x', s=hyp_p_size)
        if self.accumulated_fps is not None:
            plt.scatter(fps.loc[fps.type == 'fire'].x.values, 
                    fps.loc[fps.type == 'fire'].y.values, c='red', marker='x', s=fp_size)
            plt.scatter(fps.loc[fps.type == 'water'].x.values, 
                    fps.loc[fps.type == 'water'].y.values, c='blue', marker='x', s=fp_size)
            plt.scatter(fps.loc[fps.type == 'road'].x.values, 
                    fps.loc[fps.type == 'road'].y.values, c='black', marker='x',  s=fp_size)
