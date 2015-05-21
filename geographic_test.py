import numpy as np
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
        return fp_dist.min(axis=1)



#REAL DATA TEST (4/21/2015):
#  import pandas as pd
#  train = pd.read_csv("train_condense_wild_soil.csv")
#  train2 = train.loc[train.Wilderness_Area == 2]
#  trial = train2.reset_index()
#  gptrain = GeoParser(trial)
#  gptrain.iterate_cohorts()
#
#  in a couple runs of this process the loading generated a seemingly reasonable number of cohorts (around 5 
#    of sizes varying between 40 and 200 or so)
#  on iterating the cohorts, none of them converged, nor did the cost function even improve with any of the
#    jiggling.
#  It seems a likely next step to dig into the specifics more and see for instance if the cohort making algorithm
#    needs to be adjusted for the real data



class GeoParser():
    """ The tools to try and determine physical x, y coordinates of samples and fixed points of 
        forest cover data.  To date the fields used for input are the horizontal distances to
        fire points, water, and road
        example:
        td = ForestCoverTestData(64)
        gp = GeoParser(td.data) #data is pandas dataframe containing the horizontal distance to fire,
          water, road
          #this will automatically split the data into cohorts based on points that seem to be close
          #to each other
        gp.iterate_cohorts() #will go through each cohort of points and try to find x,y positions for them 
          #and fixed points to minimize error when compared to given distances to fixed points
        points = gp.automate_cohort_matching() #attempts to match the cohorts up by recentering/rotating/reflecting
          #so that points that are in more than one cohort end up with the same x, y coordinates 
          #(or as nearly as possible) in each.
        compare_plots(td.points, points, td.fixed_points, gp.fixed_points.loc[[0,1,2]], rotation_angle, reflection)
          #Assuming true points are known from test data.  If using real data will likely 
          #not know the true points so will be unable to make this comparison plot.
          #This will plot true x, y coordinates for samples and fixed points against the generated values
          #found.  The plot will re-center the data sets so the fire fixed point is at 0,0.  If the 
          #process worked the data should line up up to rotation or reflection which can be added to try and
          #get the points to line up in the plot.
          #reflection can be 'x', 'y' or None - defaults to None
          #the fixed point colors in the comparison plots should align, but for the sample points the colors
          #  will likely not align since they are not naturally ordered the same.
    """
    def __init__(self, data, radius=1500):
        self.data = data
        self.radius = radius
        self.cohort_threshold = 4  #cohorts smaller than this don't seem to consistently converge correctly
        self.n = len(data)
        self.center_ids = []
        self.accumulated_cohorts = None
        self.fixed_points = None
        self.points = []
        #self.good_cohorts = []
        #self.make_cohorts(radius)
        self.cost_threshold = 25
        self.init_fixed_points()

    
    def init_fixed_points(self):
        self.df_fixed_points = pd.DataFrame(columns=['x', 'y', 'type'])

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

    def point_bgfs_call(self, x=None):
        """coordinate the bgfs/gradient descent algorithm

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
    
    def fit_current_cohort(self):
        if len(self.current_cohort) < self.cohort_threshold:
            print "FAILED to fit cohort, size got too small with reductions"
            return False
        p, fp = self.init_points(self.current_cohort)
        start_time = time.clock()
        p, fp, cost = self.bgfs_automate()
        end_time = time.clock()
        total_time = end_time - start_time
        if cost <= self.cost_threshold:
            print "SUCCESS at fitting cohort of length {} cost {}, time {}".format(len(self.current_cohort), cost, total_time)
            self.points.append(p)
            self.fixed_points = fp
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
                return self.fit_current_cohort()
            else:
                print "No more new points in current cohort, abandoning attempt"
                return False


    def remove_problem_points(self, results, percentile=.75):
        cutoff = results.total.quantile(percentile)
        keep_indices = np.append(results.loc[results.total < cutoff].Id, self.data.loc[self.data.Id.isin(self.center_ids)].Id)
        self.current_cohort = self.current_cohort.loc[self.current_cohort.Id.isin(keep_indices)]

    def find_cohort(self, pid): 
        """
            pid: value of Id field of center point of cohort
        """
        water_v_threshold = 5
        ds = pd.DataFrame(np.zeros((self.n, 1)), columns=['distance'])
        locus = self.data.loc[self.data.Id == pid]
        #First filter by if water source could be same - i.e. Vertical_Distance_To_Hydrology - Elevation is same (close to same?)
        water_elev = (locus.Vertical_Distance_To_Hydrology - locus.Elevation).values[0]
        filter_indices = self.data.loc[
                (self.data.Vertical_Distance_To_Hydrology - self.data.Elevation) - water_elev < water_v_threshold].index
        fds = ds.loc[filter_indices]
        fdata = self.data.loc[filter_indices]
        fds.distance += (locus.Horizontal_Distance_To_Fire_Points.values[0] - fdata.Horizontal_Distance_To_Fire_Points)**2
        fds.distance += (locus.Horizontal_Distance_To_Roadways.values[0] - fdata.Horizontal_Distance_To_Roadways)**2
        fds.distance += (locus.Horizontal_Distance_To_Hydrology.values[0] -  fdata.Horizontal_Distance_To_Hydrology)**2
        fds.distance = np.sqrt(fds.distance)
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

    def update_values(self, points, fixed_points, alpha, x_update, y_update, fp_update):
        #Performs the basic arithmetic to update the values of the hypothesized points and fixed_points
        #points:  DataFrame of hypothesized x, y coordinates
        #fixed_points:  dict of hypothesized fixed_point coordinates
        #alpha:  learning rate (a scalar)
        #*_update:  values to update *-coordinates with
        points.x = points.x - alpha*x_update
        points.y = points.y - alpha*y_update
        for t in ['fire', 'water', 'road']:
            fpi = fixed_points[fixed_points.type == t].iloc[0].name
            fixed_points.loc[fpi, 'x'] -= alpha*fp_update[type]['x']
            fixed_points.loc[fpi, 'y'] -= alpha*fp_update[type]['y']
        return points, fixed_points

    def iterate_hypothesis(self, n, cohort, alpha=.02, p=None, fp=None, show_costs=True):
        #Iterates the gradient descent algorithm
        #Prints out cost of the hypothesized coordinates at intervals throughout the iteration
        #Returns p, fp that have resulted at end of n iterations
        #n:  Number of iterations (integer)
        #cohort:  DataFrame containing true distances of points from fixed points
        #alpha:  learning rate (positive real number
        #p:  DataFrame of hypothesized x, y coordinates.  Will initialize to random values if not supplied
        #fp: DataFrame of hypothesized fixed point coordinates.  Will initialize to random values if not supplied
        periodic_costs = []
        if p is None or fp is None:
            print "initializing points"
            p, fp = self.init_points(cohort)
        periodic_costs.append(self.cost(cohort, p, fp))
        if show_costs:
            print "Initial cost {:.3f}".format(periodic_costs[-1])
        digit_size = int(np.log10(n))
        print_mod = 10**(digit_size-1)
        for i in range(n):
            px, py, pfix = self.cost_deriv(cohort, p, fp)
            p, fp = self.update_values(p, fp, alpha, px, py, pfix)
            
            if (i+1)%print_mod == 0:
                periodic_costs.append(self.cost(cohort, p, fp))
                digits = int(np.log10(periodic_costs[-1] + 1) + 1)
                if show_costs:
                    print "Iteration {}: cost {:.3f}, digit count {}".format(i+1, periodic_costs[-1], digits)
                if periodic_costs[-1] < self.cost_threshold:
                    print "Breaking off since cost ({:.3f}) is less than threshold value ({})".format(periodic_costs[-1], self.cost_threshold)
                    break
        return p, fp, periodic_costs

    def automate_jiggle(self, data, p, fp):
        #TODO: For real data maybe need to remove some points from a cohort rather than just jiggling?
        order = self.examine_results(data, p, fp)
        indices = order.sort('total', ascending=False).head(1).index
        to_jiggle = []
        if self.worst_index == indices[0]:
            self.worst_change_count += 1
        else:
            self.best_jiggle[self.worst_change_count] += 1
            self.worst_index = indices[0]
            self.worst_change_count = 0
        about = order.loc[indices[0], ['fire', 'water', 'road']].abs().idxmin() #Which fixed point to reflect over
        for i in indices:
            if order.loc[i, ['fire', 'water', 'road']].abs().idxmin() == about:
                to_jiggle.append(i)

        self.adjust_count += 1
        if self.worst_change_count == 0:
            self.jiggle(to_jiggle, p, fp, about, 'both')
        elif self.worst_change_count == 1:
            self.jiggle(to_jiggle, p, fp, about, 'x')
        elif self.worst_change_count == 2:
            self.jiggle(to_jiggle, p, fp, about, 'y')
        else:
            print "None of the jiggling worked! :-( index {}".format(indices[0])
            print "Best jiggles {}".format(self.best_jiggle)
            print "Adjustment count {}".format(self.adjust_count)
            return False
        return True

    def jiggle(self, indices, points, fixed_points, about='fire', axis='both', rand=False):
        if rand:
            for index in indices:
                points.loc[index, 'x'] = 1000*np.random.randn(1)
                points.loc[index, 'y'] = 1000*np.random.randn(1)
        else:
            fp = fixed_points.loc[fixed_points.type == about].iloc[0]
            cx = -fp['x']
            cy = -fp['y']
            self.recenter(fixed_points, points, cx, cy)
            for index in indices:
                if axis == 'x' or axis == 'both':
                    points.loc[index, 'x'] = - points.loc[index, 'x']
                if axis == 'y' or axis == 'both':
                    points.loc[index, 'y'] = - points.loc[index, 'y']
            self.recenter(fixed_points, points, -cx, -cy)


    def automated_iteration(self, data, n=1000, alpha=2, show_costs=False):
        inc_improvement_threshold = .005 #Relative change in cost at which we wnt to adjust values before continuing
                                        #Probably makes sense for this to be different depending on n
                                        #  but if we are reasonably consistent with n can just hand tune as needed
        p, fp, costs = self.iterate_hypothesis(n, data, alpha, None, None, show_costs)
        loop_count = 1
        adjust_count = 0
        cost_change = (costs[-2] - costs[-1])/costs[-1]
        worst_index = -1
        worst_change_count = 0
        best_jiggle = [0,0,0]
        
        while costs[-1] > self.cost_threshold:
            while cost_change is None or cost_change >= inc_improvement_threshold:
                if cost_change is not None:
                    print "Doing well (change of {:.3f}).  Cost {:.3f}.  Keeping at it.".format(cost_change, costs[-1])
                loop_count += 1
                p, fp, costs = self.iterate_hypothesis(n, data, alpha, p, fp, show_costs)
                if costs[-1] <= 1:  #At this point absolute change more compelling than relative change
                    cost_change = (costs[-2] - costs[-1])
                else:
                    cost_change = (costs[-2] - costs[-1])/costs[-1]
                if costs[-1] < self.cost_threshold:
                    print "We did it!  cost: {:.3f}".format(costs[-1])
                    print "Best jiggles {}".format(best_jiggle)
                    print "Loop count {}, total iterations {}".format(loop_count, loop_count*n)
                    print "Adjustment count {}".format(adjust_count)
                    return p, fp, costs
            else:
                print "Did not change much ({:.3f}), cost {:.3f} - need to adjust".format(cost_change, costs[-1])
            order = self.examine_results(data, p, fp)
            indices = order.sort('total', ascending=False).head(1).index
            if worst_index == indices[0]:
                worst_change_count += 1
            else:
                best_jiggle[worst_change_count] += 1
                worst_index = indices[0]
                worst_change_count = 0
            about = order.loc[indices[0], ['fire', 'water', 'road']].abs().idxmin()
            adjust_count += 1
            if worst_change_count == 0:
                self.jiggle(indices, p, fp, about, 'y')
            elif worst_change_count == 1:
                self.jiggle(indices, p, fp, about, 'x')
            elif worst_change_count == 2:
                self.jiggle(indices, p, fp, about, 'both')
            else:
                print "None of the jiggling worked! :-( index {}".format(indices[0])
                print "Best jiggles {}".format(best_jiggle)
                print "Loop count {}, total iterations {}".format(loop_count, loop_count*n)
                print "Adjustment count {}".format(adjust_count)            
                return p, fp, costs
            cost_change = None
        else:
            print "Best jiggles {}".format(best_jiggle)
            print "Loop count {}, total iterations {}".format(loop_count, loop_count*n)
            print "Adjustment count {}".format(adjust_count)
            return p, fp, costs

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

    def rotate(self, points, fixed_points, angle):
        A = np.zeros((2,2))
        A[0,0] = np.cos(angle)
        A[1,0] = -np.sin(angle)
        A[0,1] = np.sin(angle)
        A[1,1] = np.cos(angle)
        return np.dot(points[['x', 'y']], A), np.dot(fixed_points[['x', 'y']], A)
        #return np.transpose(np.dot(A, np.transpose(points[['x', 'y']]))), np.dot(A, np.transpose(fixed_points[['x', 'y']]))

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
        ordering = pd.DataFrame(np.zeros((len(p), 4)), columns=['fire', 'water', 'road', 'total'])
        ordering['Id'] = ordering.index
        for i in range(len(p)):
            fire = fp[fp.type == 'fire'].iloc[0]
            water = fp[fp.type == 'water'].iloc[0]
            road = fp[fp.type == 'road'].iloc[0]
            fire_d = np.sqrt((p.x.iloc[i] - fire['x'])**2 + (p.y.iloc[i] - fire['y'])**2)
            water_d = np.sqrt((p.x.iloc[i] - water['x'])**2 + (p.y.iloc[i] - water['y'])**2)
            road_d = np.sqrt((p.x.iloc[i] - road['x'])**2 + (p.y.iloc[i] - road['y'])**2)
            true_f = cohort.Horizontal_Distance_To_Fire_Points.iloc[i]
            true_w = cohort.Horizontal_Distance_To_Hydrology.iloc[i]
            true_r = cohort.Horizontal_Distance_To_Roadways.iloc[i]
            ordering.loc[i, ['fire', 'water', 'road']] = [fire_d - true_f, water_d - true_w, road_d - true_r]
        ordering['total'] = abs(ordering['fire']) + abs(ordering['water']) + abs(ordering['road'])
        return ordering


    
# Functions to create test data and display results

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
    p_size = 60
    if inc_true:
        #plt.scatter(true_points.x, true_points.y, c=range(0, len(true_points)), marker='o', s=p_size)
        plt.scatter(true_points.x, true_points.y, c='green', marker='o', s=p_size)
        plt.scatter([true_fixed_points.loc[true_fixed_points.type == 'fire'].x], 
                [true_fixed_points.loc[true_fixed_points.type == 'fire'].y], c='red', s=fp_size)
        plt.scatter([true_fixed_points.loc[true_fixed_points.type == 'water'].x], 
                [true_fixed_points.loc[true_fixed_points.type == 'water'].y], c='blue', s=fp_size)
        plt.scatter([true_fixed_points.loc[true_fixed_points.type == 'road'].x], 
                [true_fixed_points.loc[true_fixed_points.type == 'road'].y], c='black', s=fp_size)
    if inc_hyp:
        #plt.scatter(hyp_points.x, hyp_points.y, c=range(0, len(hyp_points)), marker='x', s=p_size)
        plt.scatter(hyp_points.x, hyp_points.y, c='green', marker='x', s=p_size)
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

def compare_plots(true_p, hyp_p, true_fp, hyp_fp, rotation=0, reflection=None):
    fp_size = 450
    p_size = 60
    true_shift_x = -true_fp.loc[true_fp.type == 'fire'].iloc[1].x
    true_shift_y = -true_fp.loc[true_fp.type == 'fire'].iloc[1].y
    hyp_shift_x = -hyp_fp.loc[hyp_fp.type == 'fire'].iloc[0].x
    hyp_shift_y = -hyp_fp.loc[hyp_fp.type == 'fire'].iloc[0].y
    recenter(true_fp, true_p, true_shift_x, true_shift_y)
    recenter(hyp_fp, hyp_p, hyp_shift_x, hyp_shift_y)
    plt.figure(figsize=(12, 8))
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
    plt.scatter(pr[:, 0], pr[:, 1], c='orange', marker='x', s=p_size)
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
