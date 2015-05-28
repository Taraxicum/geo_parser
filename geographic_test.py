import numpy as np
import scipy
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import time

def init(radius):
    test = pd.read_csv("test_condense_wild_soil.csv")
    test2 = test.loc[(test.Wilderness_Area == 2)]
    return GeoParser(test2, radius)
    #train2 = train.loc[(train.Wilderness_Area == 2) & (train.Soil_type == 23)]
    #trial = train2.reset_index()
    #return GeoParser(trial, radius)
    

class ForestCoverTestData():
    """Generates test data to mimic forest cover data for reverse mapping to geographic coordinates
    it starts by randomly generating samples with physical location data, as well as fixed points 
    (fire point, water, road).  It then calculates the fields for each sample for distance to water,
    road, fire.
    Have not yet implemented a way to have more than one of a particular fixed point (e.g. two water
    sources)
    Example: td = ForestCoverTestData(32, 2) will generate a set of 32 points of test data
        with 2 sets of fixed points and with:
        td.data contains the fields as we would get from the forest cover data set
          (horizontal_distance_to_fire_points, etc.)
    """
    def __init__(self, n=64, fp=1):
        self.n = n
        self.points = pd.DataFrame(1000*np.random.randn(self.n, 2), columns=['x', 'y'])
        self.init_fixed_points(fp)
        self.find_distances()
        self.data['Id'] = self.data.index
        self.data['Vertical_Distance_To_Hydrology'] = 0
        self.data['Elevation'] = 0
    
    def init_fixed_points(self, sets):
        """
        :sets: number of sets of fixed points
        """

        vals = 2000*np.random.randn(6*sets)
        self.fixed_points = pd.DataFrame(vals.reshape((3*sets,2)), columns=['x', 'y'])
        self.fixed_points['type'] = sets*['fire', 'water', 'road']
        pass

    def find_distances(self):
        #Given points and fixed_points will create a DataFrame containing the distances from the points to the fixed points
        #This is intended to build a DataFrame with values similar to what we get from the kaggle competition from
        #  the points and fixed_point coordinates that we generated for testing
        #points:  DataFrame of x, y coordinates
        #fixed_points:  dict of fixed point coordinates (fire, water, road)
        self.data = pd.DataFrame(np.zeros((len(self.points), 3)), columns=['Horizontal_Distance_To_Fire_Points',
                                                               'Horizontal_Distance_To_Hydrology',
                                                               'Horizontal_Distance_To_Roadways'])
        self.data['Horizontal_Distance_To_Fire_Points'] = self.distance_to_fp('fire')
        self.data['Horizontal_Distance_To_Hydrology'] = self.distance_to_fp('water')
        self.data['Horizontal_Distance_To_Roadways'] =  self.distance_to_fp('road')

    def distance_to_fp(self, fp_type):
        """
        return a Series of length n (n is the number of test points)
            where the kth value is the minimum distance of the kth test point
            to a fixed point of type fp_type
        """
        fp_numb = len(self.fixed_points.loc[self.fixed_points.type == fp_type])
        fp_dist = pd.DataFrame(np.zeros((len(self.points), fp_numb)))
        count = 0
        for i, fp in self.fixed_points.loc[self.fixed_points.type == fp_type].iterrows():
            fp_dist[count] = np.sqrt((self.points.x -  fp.x)**2 + (self.points.y - fp.y)**2)
            count += 1
        self.points[fp_type + '_index'] = fp_dist.idxmin(axis=1)
        
        return fp_dist.min(axis=1)


class GeoParser():
    """ The tools to try and determine physical x, y coordinates of samples and fixed points of 
        forest cover data.  To date the fields used for input are the horizontal distances to
        fire points, water, and road
        TODO update process description as process stabilizes
    """
    def __init__(self, data, radius=1500):
        self.data = data
        self.radius = radius
        self.norm_radius = .5 #TODO Testing this value, may want to adjust
        self.cohort_threshold = 4  #cohorts smaller than this don't seem to consistently converge correctly
        self.n = len(data)
        self.center_ids = []
        self.accumulated_cohorts = None
        self.fixed_points = None
        self.points = []
        #self.good_cohorts = []
        #self.make_cohorts(radius)
        self.cost_threshold = 1
        self.tested_centers = {}
        self.good_count = 0
        self.good_points = []
        self.good_fp = []
        self.init_fixed_points()

    
    def init_fixed_points(self):
        self.df_fixed_points = pd.DataFrame(columns=['x', 'y', 'type'])

    
    def fp_cost(self, x):
        """Cost function for fitting fixed points to the points in self.current_cohort which
        must have already been fit to some x, y coordinates

        :x: list of fixed points in order fire_x, fire_y, water_x, water_y, road_x, road_y
        :returns: non-negative real cost
        """
        points = self.points_to_vector(self.current_cohort)
        return self.bgfs_cost(np.append(x, points))
    
    
    def fp_gradient(self, x):
        """Gradient of the fp_cost function.  Similar to self.bgfs_gradient, but treats points
        as constants, and fixed points as variables.
        self.current_cohort must have points defined with x, y coordinates

        :x: list of fixed points in order fire_x, fire_y, water_x, water_y, road_x, road_y
        :returns: gradient vector in same format as x
        """
        m = len(self.current_cohort)
        points = self.points_to_vector(self.current_cohort)
        full_vect = np.append(x, points)
        grad_components = self.gradient_parts(full_vect)
        grad_sum = grad_components[['main_fire_x', 'main_fire_y', 
            'main_water_x', 'main_water_y', 'main_road_x', 'main_road_y']].sum(axis=0)
        grad = np.zeros(6)
        grad[0] = grad_sum['main_fire_x']/(2*m)
        grad[1] = grad_sum['main_fire_y']/(2*m)
        grad[2] = grad_sum['main_water_x']/(2*m)
        grad[3] = grad_sum['main_water_y']/(2*m)
        grad[4] = grad_sum['main_road_x']/(2*m)
        grad[5] = grad_sum['main_road_y']/(2*m)
        return grad
    
    def fit_points_to_fp(self):
        """
        The goal here is if there is a set of 3 fixed points that are believed to be good,
        see what sample points can fit those fixed points well.
        Uses the fixed points in self.fixed_points
        Uses the points in self.data
        return: No return value, but updates accumulated_cohorts with the points that fit well
        """
        #do using gradient descent (bgfs)
        #Want to make a vector of points (easy - just new data frame x, y length of self.data)
        #Cost/Gradient functions very similar to bgfs_cost/bgfs_gradient, but treat fixed points
        #  as constants.  Should be simpler than bgfs_cost/bgfs_gradient
        #examine_results of the resulting points, use the points that fall within an
        #  acceptable threshold.
        threshold = .5
        p = self.point_bgfs_call()
        results = self.examine_results(self.data, p, self.fixed_points)
        keep = results.loc[results.total < threshold].Id
        good = p.loc[p.index.isin(keep)]

    def fp_bgfs_call(self, x=None):
        """coordinate the bgfs/gradient descent algorithm for fixed points (points constant)
        self.current_cohort should contain the points to fit the fixed points against

        :x: vector of hypothesized fixed points [fire_x, fire_y, water_x, water_y, road_x, road_y]
        """
        if x is None:
            x = np.random.randn(6)
        print "STARTING fp_bgfs_call"
        start_time = time.clock()
        out = opt.fmin_bfgs(self.fp_cost, x)#, self.fp_gradient, disp=True)
        fp = self.fp_from_vector(out)
        end_time = time.clock()
        total_time = end_time - start_time
        print "END point_bgfs_call total time: {}".format(total_time)
        return fp

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
            print "PROBLEM: GeographicParser.point_gradient(x), len(x) should be 2*len(self.data)"
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
        """Attmept to find fixed points for center c

        :center: should be index of a point in self.data
        :returns: finds cohort about center, fits, records cost of fit in self.tested_centers
        if cost is less than threshold, records fixed points
        """
        if center is None:
            center = self.data.index[np.random.randint(len(self.data))]
        print "Attempting fixed points for {}".format(center)
        self.current_cohort = self.data.loc[self.find_normed_cohort(center, 6, test)]
        self.fit_current_cohort(center)

    
    def chain_progressive_cohorts(self, n=2):
        if n == 0:
            return

        if self.progressive_cohorts():
            if self.accumulated_cohorts is None:
                self.add_fixed_points(self.fixed_points)
                self.accumulated_cohorts = pd.DataFrame(self.current_cohort)
            else:
                self.match_cohort()
                #self.accumulated_cohorts = self.accumulated_cohorts.append(self.current_cohort, ignore_index = True)
            print "Accumulated {} points".format(len(self.accumulated_cohorts))
        else:
            print "Trouble completing progress_cohorts() with {} iterations left".format(n -1)
            return
        if len(self.accumulated_cohorts) < len(self.data):
            self.chain_progressive_cohorts(n - 1)
    
    def set_magnitudes(self):
        """
            Set accumulated_cohorts['magnitude'] field to the
            point distance from the mean.
            The goal here is to make it easier to find points towards
            the edge of those already incorporated so we can use them
            for centers of new cohorts to add in
        """
        mean_x = self.accumulated_cohorts.loc[self.data.Id.isin(self.center_ids)].x.mean(axis=0)
        mean_y = self.accumulated_cohorts.loc[self.data.Id.isin(self.center_ids)].y.mean(axis=0)
        self.accumulated_cohorts['magnitude'] = np.sqrt(
                (self.accumulated_cohorts.x - mean_x)**2 + 
                (self.accumulated_cohorts.y - mean_y)**2)

    def pick_center_points(self, n=10):
        """
            Pick list of options for center points.
            If cohorts have been accumulated, will return points from that accumulation
            If no cohorts have been accumulated will pick randomly from the data set
            if possible returns Series the Ids of n points to try as new cohort center
        """
        if self.accumulated_cohorts is None:
            return self.data.iloc[np.random.randint(len(self.data), size=n)].Id
        else:
            #self.set_magnitudes()
            #options = self.accumulated_cohorts.sort('magnitude', ascending=False).Id.head(n)
            #Try Random:
            length = len(self.accumulated_cohorts)
            options = self.accumulated_cohorts.iloc[np.random.randint(length, size=n)].Id
            return options
    
    def check_good_cohort(self, indices):
        """
            Have to be a minimum number of indices to bother making a cohort
            If so, fit the cohort and if that is successful, make sure the
            resulting cohort actually adds at least one point to the accumulated cohorts
        """
        if len(indices) > self.cohort_threshold:
            self.current_cohort = self.data.loc[indices]
            if self.accumulated_cohorts is None:
                amount_new = len(self.current_cohort)
            else:
                amount_new = len(set(self.current_cohort.Id) - set(self.accumulated_cohorts.Id))
            if (amount_new > 0) and self.fit_current_cohort():
                if self.accumulated_cohorts is None:
                    return True
                overlap = len(set(self.current_cohort.Id) - set(self.accumulated_cohorts.Id))
                print "New cohort adds {} points".format(overlap)
                if overlap > 0:
                    return True
        return False

    def progressive_cohorts(self):
        indices = []
        attempt_count = 0
        options = self.pick_center_points(30)
        for point in options:
            if self.check_good_cohort(indices):
                self.center_ids.append(point)
                return True
            #point = self.data.iloc[np.random.randint(len(self.data))].Id
            indices = self.find_cohort(point)
            print "Found {} points for cohort - center {} - in attempt {}".format(len(indices), point, attempt_count + 1)
            attempt_count += 1
        if self.check_good_cohort(indices):
            self.center_ids.append(point)
            return True
        else:
            print "Attempt to find new cohort failed after {} attempts".format(len(options))
            return False
     
    def init_points(self, cohort):
            #Initializes set of hypothesized points and fixed points for iterating 
            #cohort:  DataFrame containing the true distances of the points to the 
            points = pd.DataFrame(100*np.random.randn(len(cohort), 2), columns=['x', 'y'])
            vals = 1000*np.random.randn(6)
            fixed_points = pd.DataFrame(vals.reshape((3,2)), columns=['x', 'y'])
            fixed_points['type'] = ['fire', 'water', 'road']
            return points, fixed_points
    
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


    def remove_problem_points(self, results, percentile=.75):
        cutoff = results.total.quantile(percentile)
        keep_indices = np.append(results.loc[results.total < cutoff].Id, self.data.loc[self.data.Id.isin(self.center_ids)].Id)
        self.current_cohort = self.current_cohort.loc[self.current_cohort.Id.isin(keep_indices)]

    def norm_distance_fields(self):
        fp = self.data.Horizontal_Distance_To_Fire_Points
        self.data['normed_fire'] = (fp - fp.mean())/fp.std()
        fp = self.data.Horizontal_Distance_To_Hydrology
        self.data['normed_water'] = (fp - fp.mean())/fp.std()
        fp = self.data.Horizontal_Distance_To_Roadways
        self.data['normed_road'] = (fp - fp.mean())/fp.std()

    def find_normed_cohort(self, pid, best_n = 0, test=None):
        """
            pid: value of Id field of center point of cohort
        """
        water_v_threshold = 5
        fds = pd.DataFrame(columns=['distance'])
        locus = self.data.loc[self.data.Id == pid]
        #First filter by if water source could be same - i.e. Vertical_Distance_To_Hydrology - Elevation is same (close to same?)
        water_elev = (locus.Vertical_Distance_To_Hydrology - locus.Elevation).values[0]
        filter_indices = self.data.loc[
                (self.data.Vertical_Distance_To_Hydrology - self.data.Elevation) - water_elev < water_v_threshold].index
        
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
            return fds.loc[fds.distance < self.norm_radius].index
        elif best_n > 0:
            return fds.sort('distance').head(best_n).index
        else:
            print "GeoParser.find_normed_cohort: Invalid value for best_n should be non negative integer, got {}".format(best_n)
            return
            
    
    def find_cohort(self, pid, test=None): 
        """
            pid: value of Id field of center point of cohort
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

    def add_fixed_points(self, fp_set):
        #fp_set should be set of fire, water, road fixed points
        fixed_point_difference_threshold = 10 #if fixed points of same type are within difference threshold of
            #each other, treat as same point
        for i in fp_set.index:
            same = False
            #TODO check first not 'the same' as an existing
            if not self.df_fixed_points.loc[self.df_fixed_points.type == fp_set.loc[i]['type']].empty:
                for fp in self.df_fixed_points.loc[self.df_fixed_points.type == fp_set.loc[i]['type']].iterrows():
                    if self.distance_xy(fp[1], fp_set.loc[i]) < fixed_point_difference_threshold:
                        same = True
                        break
            if not same:
                self.df_fixed_points.loc[len(self.df_fixed_points), 
                    ['x', 'y', 'type']] = [fp_set.loc[i]['x'], fp_set.loc[i]['y'], fp_set.loc[i]['type']]


    def match_cohort(self):
        #cohort should have at least 3 overlapping points to be matched up
        #theoretically the overlap could include the fixed points, but in practice that might be a little
        #tricky since additional checking would need to be done to ensure the fixed points were the same
        joined_points = self.accumulated_cohorts
        if len(set(joined_points.index) & set(self.current_cohort.index)) >= 3:
            self.align_cohorts(joined_points, self.current_cohort, self.fixed_points)
            joined_points = pd.concat([joined_points, 
                self.current_cohort.loc[set(self.current_cohort.index) - set(joined_points.index)]])
            self.add_fixed_points(self.fixed_points)
            self.accumulated_cohorts = joined_points
        else:
            print "Not enough overlap between accumulated cohorts and current cohort?!" #Should not happen

    def bgfs_automate(self, p=None, fp=None):
        if (p is None) or (fp is None):
            p, fp = self.bgfs_call()
        else:
            x = self.points_fp_to_vector(p, fp)
            cost = self.cost(self.current_cohort, p, fp)
            p, fp = self.bgfs_call(x)
        #TODO implement mechanism for correcting poor fits
        cost = self.cost(self.current_cohort, p, fp)
        return p, fp, cost
    
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
        fire_d = (self.dist(points, fixed_points[fixed_points.type == 'fire']) - self.current_cohort.Horizontal_Distance_To_Fire_Points.values)**2
        water_d = (self.dist(points, fixed_points[fixed_points.type == 'water']) - self.current_cohort.Horizontal_Distance_To_Hydrology.values)**2
        road_d = (self.dist(points, fixed_points[fixed_points.type == 'road']) - self.current_cohort.Horizontal_Distance_To_Roadways.values)**2
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
    
    def distance_xy(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


    def dist(self, points, fp):
        #Distance between points and fixed point (fire, water, road).
        #points: should be a DataFrame 
        #fp: should be a single fixed point object (e.g. has x, y, type properties)
        fpi = fp.iloc[0].name
        return np.sqrt((points.x - fp.loc[fpi, 'x'])**2 + (points.y - fp.loc[fpi, 'y'])**2)

    def partial(self, cohort_d, points, fp):
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
        
    def points_fp_to_vector(self, points, fp):
        return np.concatenate([self.fp_to_vector(fp), self.points_to_vector(points)])

    def points_to_vector(self, points):
        return np.concatenate([points.x.values, points.y.values])

    def points_from_vector(self, x):
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


    def fp_to_vector(self, fp):
        x = np.zeros(6)
        x[0] = fp.loc[fp.type == 'fire'].iloc[0]['x']
        x[1] = fp.loc[fp.type == 'fire'].iloc[0]['y']
        x[2] = fp.loc[fp.type == 'water'].iloc[0]['x']
        x[3] = fp.loc[fp.type == 'water'].iloc[0]['y']
        x[4] = fp.loc[fp.type == 'road'].iloc[0]['x']
        x[5] = fp.loc[fp.type == 'road'].iloc[0]['y']
        return x

    def fp_from_vector(self, x):
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
        X = secondary.loc[overlap, ['x', 'y']].values
        Y = primary.loc[overlap, ['x', 'y']].values
        transform = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), Y))
        secondary[['x', 'y']] = np.dot(secondary[['x', 'y']], transform)
        #Need to rotate fixed points as well
        secondary_fp[['x', 'y']] = np.dot(secondary_fp[['x', 'y']], transform)
        self.recenter(None, primary, shift_x, shift_y)
        self.recenter(secondary_fp, secondary, shift_x, shift_y)
        return True
    
    def examine_results(self, cohort, p, fp):
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

def match_on_fp(cohorts, fps):
    if len(cohorts) != len(fps):
        print("Problem in match_on_fp():  cohorts and fps should be lists of same length instead got "
                "len(cohort) {}, len(fps) {}".format(len(cohorts), len(fps)))
        return None
    distance_threshold = 50
    
    #first remove duplicate point sets
    point_set = set()
    count = 0
    working_cohorts = []
    working_fps = []
    for i in range(len(cohorts)):
        point_set = point_set.union(set(cohorts[i].index))
        if len(point_set) != count: #added new points
            working_cohorts.append(cohorts[i])
            working_fps.append(fps[i])
            count = len(point_set)
    accumulated_cohort = pd.DataFrame(columns = working_cohorts[0].columns)
    accumulated_fps = pd.DataFrame(columns=working_fps[0].columns)
    accumulated_cohort = accumulated_cohort.append(working_cohorts[0])
    accumulated_fps = accumulated_fps.append(working_fps[0])
    for i in range(1,len(working_cohorts)):
        align_pair_on_fp(working_cohorts[i], working_fps[i], accumulated_cohort, accumulated_fps)

def align_pair_on_fp(cohort1, fp1, cohort2, fp2):
    """
    align cohorts using points and fixed points to line up.
    fixed points may contain more than 1 of any type but must have at least one of each
    (e.g. could have 3 water points, 1 fire point, 2 road points)
    returns cohort, fp that is best fit of the pairs
    """
    #Try to align on fixed points first
    distance_threshold = 50 #if fixed point pairs are less than threshold difference in distance,
        #treat as same pair
    best_fits = {}
    for i in range(len(fp1)):
        for j in range(i + 1, len(fp1)):
            (dist, pair2) = fixed_point_match(fp1.iloc[i], fp1.iloc[j], fp2)
            if dist < distance_threshold:
                best_fits[(fp1.iloc[i].name, fp1.iloc[j].name)] = {'cost': dist, 'match': pair2}
    if best_fits != {}:
        array1 = []
        array2 = []
        for k in best_fits.keys():
            array1.append(fp1.loc[k[0], ['x', 'y']].values)
            array1.append(fp1.loc[k[1], ['x', 'y']].values)
            array2.append(fp2.loc[best_fits[k]['match'][0], ['x', 'y']].values)
            array2.append(fp2.loc[best_fits[k]['match'][1], ['x', 'y']].values)
            #p1a = fp1.loc[k[0]]; p1b = fp1.loc[k[1]]
            #p2a = fp2.loc[best_fits[k]['match'][0]]; p2b = fp2.loc[best_fits[k]['match'][1]]
        #fit_plot(p1a, p1b, p2a, p2b, cohort1, cohort2, fp1, fp2)
        overlap1 = pd.DataFrame(array1, columns=['x', 'y'])
        overlap2 = pd.DataFrame(array2, columns=['x', 'y'])
        fit_plot(overlap1, overlap2, cohort1, cohort2, fp1, fp2)


def distance_xy(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def fixed_point_match(fp1a, fp1b, fp_set):
    dist1 = distance_xy(fp1a, fp1b)
    fp2a = fp_set[fp_set.type == fp1a.type]
    fp2b = fp_set[fp_set.type == fp1b.type]
    closest = -1
    best = None
    for i, a in fp2a.iterrows():
        for j, b in fp2b.iterrows():
            dist2 = distance_xy(a, b)
            val = abs(dist1 - dist2)
            if closest < 0 or val < closest:
                closest = val
                best = (a.name, b.name)
    return closest, best


# Functions to create test data and display results

def fit_plot(overlap1, overlap2, points1, points2, fp1, fp2):
    #shift so overlap centroids are at the origin
    centroid1 = overlap1.mean()
    centroid2 = overlap2.mean()
    overlap1 = overlap1 - centroid1
    overlap2 = overlap2 - centroid2
    points1[['x', 'y']] -= centroid1
    points2[['x', 'y']] -= centroid2
    fp1[['x', 'y']] -= centroid1
    fp2[['x', 'y']] -= centroid2
    
    #try with reflection:
    #primary[['x']] = -primary[['x']]
    #points1[['x']] = -points1[['x']]
    #fp1[['x']] = -fp1[['x']]
    #Use svd rigid transformation best fit methodology as described at http://nghiaho.com/?page_id=671
    X = overlap2[['x', 'y']].values
    Y = overlap1[['x', 'y']].values
    H = np.dot(X.T, Y)
    U, S, V = np.linalg.svd(H)
    rot = np.dot(U, V.T)

    points2[['x', 'y']] = np.dot(points2[['x', 'y']], rot)
    fp2[['x', 'y']] = np.dot(fp2[['x', 'y']], rot)
    compare_plots(points1, points2, fp1, fp2)


def plot_hypothesis_3d(points):
    plot3d(points.x, points.y, points.Elevation, points)
    plt.show()

def plot_hypothesis(points, fixed_points, center_ids, plot_all_fp=True):
    """Plot x, y coordinates hypothesized from our fitting process of
    sample points and fixed points.

    :points: DataFrame of sample points including x, y fields
    :fixed_points: DataFrame of fixed points (fire, water, road) inc. x, y
    :center_ids: Id values of centers of cohorts
    :plot_all_fp:  Plot all fixed points, or just first in list
    :returns: Displays data in matplotlib plot
    """
    if plot_all_fp:
        fire_x = fixed_points.loc[fixed_points.type == 'fire'].x
        fire_y = fixed_points.loc[fixed_points.type == 'fire'].y
        water_x = fixed_points.loc[fixed_points.type == 'water'].x
        water_y = fixed_points.loc[fixed_points.type == 'water'].y
        road_x = fixed_points.loc[fixed_points.type == 'road'].x
        road_y = fixed_points.loc[fixed_points.type == 'road'].y
    else:
        fire_x = fixed_points.loc[fixed_points.type == 'fire'].iloc[0].x
        fire_y = fixed_points.loc[fixed_points.type == 'fire'].iloc[0].y
        water_x = fixed_points.loc[fixed_points.type == 'water'].iloc[0].x
        water_y = fixed_points.loc[fixed_points.type == 'water'].iloc[0].y
        road_x = fixed_points.loc[fixed_points.type == 'road'].iloc[0].x
        road_y = fixed_points.loc[fixed_points.type == 'road'].iloc[0].y
    centers = points[points.Id.isin(center_ids)]
    plt.scatter(points.x, points.y, c='yellow', marker='x', s=60)
    plt.scatter(centers.x, centers.y, c='green', marker='o', s=160)
    plt.scatter([fire_x], 
            [fire_y], c='red', marker='x', s=250)
    plt.scatter([water_x], 
            [water_y], c='blue', marker='x',  s=250)
    plt.scatter([road_x], 
            [road_y], c='black', marker='x',  s=250)
    plt.show()
    

def plot_results(true_points, hyp_points, true_fixed_points, hyp_fixed_points, inc_true=True, inc_hyp=True):
    """
      Plot the hypothesized points and fixed points as well as the true points and fixed points.
      The true values will plot as circles, the hypothesized as x.
      The fixed points will be larger with same shape scheme and red for fire, blue for water, black for road
    """
    fp_size = 450
    true_p_size = 60
    hyp_p_size = 150

    if inc_true:
        #plt.scatter(true_points.x, true_points.y, c=range(0, len(true_points)), marker='o', s=p_size)
        plt.scatter(true_points.x, true_points.y, c='green', marker='o', s=true_p_size)
        plt.scatter(true_fixed_points.loc[true_fixed_points.type == 'fire'].x, 
                true_fixed_points.loc[true_fixed_points.type == 'fire'].y, c='red', s=fp_size)
        plt.scatter(true_fixed_points.loc[true_fixed_points.type == 'water'].x, 
                true_fixed_points.loc[true_fixed_points.type == 'water'].y, c='blue', s=fp_size)
        plt.scatter(true_fixed_points.loc[true_fixed_points.type == 'road'].x, 
                true_fixed_points.loc[true_fixed_points.type == 'road'].y, c='black', s=fp_size)
    if inc_hyp:
        #plt.scatter(hyp_points.x, hyp_points.y, c=range(0, len(hyp_points)), marker='x', s=p_size)
        plt.scatter(hyp_points.x, hyp_points.y, c='green', marker='x', s=hyp_p_size)
        plt.scatter([hyp_fixed_points.loc[hyp_fixed_points.type == 'fire'].x], 
                [hyp_fixed_points.loc[hyp_fixed_points.type == 'fire'].y], c='red', marker='x', s=fp_size)
        plt.scatter([hyp_fixed_points.loc[hyp_fixed_points.type == 'water'].x], 
                [hyp_fixed_points.loc[hyp_fixed_points.type == 'water'].y], c='blue', marker='x',  s=fp_size)
        plt.scatter([hyp_fixed_points.loc[hyp_fixed_points.type == 'road'].x], 
                [hyp_fixed_points.loc[hyp_fixed_points.type == 'road'].y], c='black', marker='x',  s=fp_size)
    plt.show()


def rotate(points, fixed_points, angle):
    A = np.zeros((2,2))
    A[0,0] = np.cos(angle)
    A[1,0] = -np.sin(angle)
    A[0,1] = np.sin(angle)
    A[1,1] = np.cos(angle)
    
    return np.dot(points[['x', 'y']], A), np.dot(fixed_points[['x', 'y']], A)

def recenter(fixed_points, points, x_amount, y_amount):
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

def compare_plots(true_p, hyp_p, true_fp, hyp_fp, rotation=0, reflection=None, fire_i=0, title=None, xlabel=None, ylabel=None):
    fp_size = 450
    true_p_size = 60
    hyp_p_size = 150
    true_shift_x = -true_fp.loc[true_fp.type == 'fire'].iloc[fire_i].x
    true_shift_y = -true_fp.loc[true_fp.type == 'fire'].iloc[fire_i].y
    hyp_shift_x = -hyp_fp.loc[hyp_fp.type == 'fire'].iloc[0].x
    hyp_shift_y = -hyp_fp.loc[hyp_fp.type == 'fire'].iloc[0].y
    recenter(true_fp, true_p, true_shift_x, true_shift_y)
    recenter(hyp_fp, hyp_p, hyp_shift_x, hyp_shift_y)
    plt.figure(figsize=(12, 8))
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    pr, fpr = rotate(hyp_p, hyp_fp, rotation)
#    maxval = max([max(abs(fpr)), max(abs(pr))
    plt.axis('equal')
    # negative x-values to make a reflection over the y-axis
    if reflection is not None and reflection.lower() == 'y':
        pr[:,0] = -pr[:,0]
        fpr[:,0] = -fpr[:,0]
    elif reflection is not None and reflection.lower() == 'x':
        pr[:,1] = -pr[:,1]
        fpr[:,1] = -fpr[:,1]
    #plt.scatter(pr[:, 0], pr[:, 1], c=range(0, len(pr[:,0])), marker='x', s=60)
    plt.scatter(pr[:, 0], pr[:, 1], c='orange', marker='x', s=hyp_p_size)
    plt.scatter(fpr[0,0], fpr[0, 1], c='red', marker='x', s=fp_size)
    plt.scatter(fpr[1,0], fpr[1, 1], c='blue', marker='x', s=fp_size)
    plt.scatter(fpr[2,0], fpr[2, 1], c='black', marker='x', s=fp_size)
    plot_results(true_p, hyp_p, true_fp, hyp_fp, inc_hyp=False)
    recenter(true_fp, true_p, -true_shift_x, -true_shift_y)
    recenter(hyp_fp, hyp_p, -hyp_shift_x, -hyp_shift_y)

#plt.scatter(df.Horizontal_Distance_To_Hydrology, 
#            df.Horizontal_Distance_To_Roadways, c='blue')
#plt.scatter(df.loc[cohorts[i].index].Horizontal_Distance_To_Hydrology, 
#            df.loc[cohorts[i].index].Horizontal_Distance_To_Roadways, c='red')

#plt.scatter(df.loc[t1].Horizontal_Distance_To_Hydrology, df.loc[t1].Horizontal_Distance_To_Roadways, s=200, c='red')

def plot3d(c1, c2, c3, dataset, cts=None):
    threedee = plt.figure(figsize=(16,8)).gca(projection='3d')
    length = len(c1)
    if length < 100:
        size = 50
    elif length < 1000:
        size = 15
    else:
        size = 5
    if 'Cover_Type' in dataset:
        cover = dataset.Cover_Type.values
        ct_set = set(cover)
        cm = plt.get_cmap()
        cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=max(ct_set))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        scatter_proxy = []
        labels = []
        cols = ['red', 'blue', 'green', 'red', 'blue', 'green', 'red']
        for i in ct_set:
            scatter_proxy.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=scalarMap.to_rgba(i), marker = 'o'))
            #labels.append(COVER[i-1])
        threedee.legend(scatter_proxy, labels, numpoints = 1)
        plt.scatter(c1, c2, zs=c3, norm=cNorm, c=cover, s=size)
    else:
        cover = 'blue'
        plt.scatter(c1, c2, zs=c3, c=cover, s=size)
        
    #classes = COVER
    #plt.title(title)
