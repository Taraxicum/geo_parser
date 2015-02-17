import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt

class ForestCoverTestData():
    def __init__(self, n=64):
      self.n = n
      self.points = pd.DataFrame(1000*np.random.randn(self.n, 2), columns=['x', 'y'])
      vals = 1000*np.random.randn(6)
      self.fixed_points = {'fire': {'x': vals[0], 'y': vals[1]}, 'water': {'x': vals[2], 'y': vals[3]},
                      'road': {'x': vals[4], 'y': vals[5]}}
      self.find_distances()

    def find_distances(self):
        #Given points and fixed_points will create a DataFrame containing the distances from the points to the fixed points
        #This is intended to build a DataFrame with values similar to what we get from the kaggle competition from
        #  the points and fixed_point coordinates that we generated for testing
        #points:  DataFrame of x, y coordinates
        #fixed_points:  dict of fixed point coordinates (fire, water, road)
        self.data = pd.DataFrame(np.zeros((len(self.points), 3)), columns=['Horizontal_Distance_To_Fire_Points',
                                                               'Horizontal_Distance_To_Hydrology',
                                                               'Horizontal_Distance_To_Roadways'])
        self.data['Horizontal_Distance_To_Fire_Points'] = np.sqrt((self.points.x - self.fixed_points['fire']['x'])**2 + 
                                                           (self.points.y - self.fixed_points['fire']['y'])**2)
        self.data['Horizontal_Distance_To_Hydrology'] = np.sqrt((self.points.x - self.fixed_points['water']['x'])**2 + 
                                                         (self.points.y - self.fixed_points['water']['y'])**2)
        self.data['Horizontal_Distance_To_Roadways'] = np.sqrt((self.points.x - self.fixed_points['road']['x'])**2 + 
                                                        (self.points.y - self.fixed_points['road']['y'])**2)


class GeoParser():
    def __init__(self, data):
        self.data = data
        self.cohort_threshold = 4  #cohorts smaller than this don't seem to consistently converge correctly
        self.n = len(data)
        self.fixed_points = []
        self.points = []
        self.make_cohorts(1500)

    
    #Make/filter cohorts -> map cohorts to coordinates -> match coordinates up for overlapping cohorts -> done?

    def iterate_cohorts(self):
        for c in self.cohorts:
            self.current_cohort = c
            #p, fp, costs = self.automated_iteration(c, 1000, 1.5, False)
            p, fp, cost = self.bgfs_automate()
            #if costs[-1] <= .5:
            if cost <= .5:
                self.points.append(p)
                self.fixed_points.append(fp)
            else:
                print "Failed to find good fit for cohort: {}".format(c)
                #print "Failed first try, try again with smaller alpha"
                #p, fp, costs = self.automated_iteration(c, 1000, .3, False)
                #if costs[-1] <= .5:
                    #self.points.append(p)
                    #self.fixed_points.append(fp)
                #else:
                    #print "STILL FAILED TO FIND GOOD SOLUTION FOR COHORT :("

    def init_points(self, cohort):
        #Initializes set of hypothesized points and fixed points for iterating our gradient descent algorithm
        #cohort:  DataFrame containing the true distances of the points to the fixed points
        points = pd.DataFrame(100*np.random.randn(len(cohort), 2), columns=['x', 'y'])
        vals = 1000*np.random.randn(6)
        fixed_points = {'fire': {'x': vals[0], 'y':vals[1]}, 'water': {'x': vals[2], 'y':vals[3]}, 
                        'road': {'x': vals[4], 'y':vals[5]}}
        return points, fixed_points
    
    def find_cohort(self, pix, radius): 
        ds = pd.DataFrame(np.zeros((self.n, 1)), columns=['distance'])
        for i in range(self.n):
            ds.loc[i, 'distance'] = self.distance_3d(self.data.loc[pix], self.data.loc[i])
        return ds.loc[ds.distance < radius]

    def make_cohorts(self, radius):
        #From testing, it appears that having cohorts of at least size 4 is desirable, I did have one example
        #  at size 4 that failed to converge nicely, but several other examples worked quite well.
        #Size 3 had some tests that worked great and some that did converged to a non-true solution
        self.cohorts = []
        indexes = []
        remaining = self.data.index
        while len(remaining) > 0:
            p = remaining[np.random.randint(len(remaining))]
            indexes.append(self.find_cohort(p, radius))
            remaining = remaining - indexes[-1].index
        print len(indexes)
        for c in indexes:
            if len(c) > self.cohort_threshold:
                self.cohorts.append(self.data.loc[c.index])
                print "Cohort {}, length {}".format(len(self.cohorts) - 1, len(c))

    def bgfs_automate(self, p=None, fp=None):
        self.current_cohort = self.cohorts[0] #FIXME this is just set here for testing
        cost_threshold = .5  #Cost at which we call it good enough
        if (p is None) or (fp is None):
            p, fp = self.bgfs_call()
        x = self.points_fp_to_vector(p, fp)
        cost = self.cost(self.current_cohort, p, fp)
        self.adjust_count = 0
        self.worst_index = -1
        self.worst_change_count = 0
        self.best_jiggle = [0,0,0]
        while (cost > cost_threshold) and (self.worst_change_count < 3):
            self.automate_jiggle(self.current_cohort, p, fp)
            x = self.points_fp_to_vector(p, fp)
            p, fp = self.bgfs_call(x)
            x = self.points_fp_to_vector(p, fp)
            cost = self.cost(self.current_cohort, p, fp)
        return p, fp, cost
    
    def bgfs_call(self, x = None):
        if x is None:
            x = np.random.randn(len(self.current_cohort)*2 + 6)
        out = opt.fmin_bfgs(self.bgfs_cost, x, self.bgfs_gradient)
        fp = self.fp_from_vector(out[0:6])
        points = self.points_from_vector(out[6:])
        return points, fp
    
    def bgfs_cost(self, x):
        #x should have first 6 arguments be fire x, y, water x, y, road x,y, then hypothesized x-vals then y-vals
        fixed_points = self.fp_from_vector(x[0:6])
        points = self.points_from_vector(x[6:])
        fire_d = (self.dist(points, fixed_points['fire']) - self.current_cohort.Horizontal_Distance_To_Fire_Points.values)**2
        water_d = (self.dist(points, fixed_points['water']) - self.current_cohort.Horizontal_Distance_To_Hydrology.values)**2
        road_d = (self.dist(points, fixed_points['road']) - self.current_cohort.Horizontal_Distance_To_Roadways.values)**2
        return 1.0/(2*len(self.current_cohort))*(fire_d.sum() + water_d.sum() + road_d.sum())
    
    def bgfs_gradient(self, x):
        #x should have first 6 arguments be fire x, y, water x, y, road x,y, then hypothesized x-vals then y-vals
        fp = self.fp_from_vector(x[0:6])
        points = self.points_from_vector(x[6:])
        px, py, fixed = self.cost_deriv(self.current_cohort, points, fp)
        fv = self.fp_to_vector(fixed)
        return np.concatenate([fv, px.values, py.values])
    
    def cost(self, cohort, points, fixed_points):
        #RETURNS non-negative real number
        #cohort of points we are trying to map to a 2d representation that fits the data
        #points are the x, y coordinates of the hypothesized points (should be len(cohort) of them)
        #fixed_points x, y coordinates of the hypothesized fire, water, road locations
        #This is not the exact same cost function the derivative of the cost function uses - this 
        #  one uses square root to make the values a little easier to think of as an average error
        #  but would unnecessarily complicate the derivative
        fire_d = np.sqrt((self.dist(points, fixed_points['fire']) - cohort.Horizontal_Distance_To_Fire_Points.values)**2)
        water_d = np.sqrt((self.dist(points, fixed_points['water']) - cohort.Horizontal_Distance_To_Hydrology.values)**2)
        road_d = np.sqrt((self.dist(points, fixed_points['road']) - cohort.Horizontal_Distance_To_Roadways.values)**2)
        return 1.0/(2*len(cohort))*(fire_d.sum() + water_d.sum() + road_d.sum())
    
    def distance_3d(self, p1, p2):
        return np.sqrt((p1['Horizontal_Distance_To_Fire_Points'] - p2['Horizontal_Distance_To_Fire_Points'])**2 +
                       (p1['Horizontal_Distance_To_Hydrology'] - p2['Horizontal_Distance_To_Hydrology'])**2 +
                       (p1['Horizontal_Distance_To_Roadways'] - p2['Horizontal_Distance_To_Roadways'])**2)

    def dist(self, points, fp):
        #Distance between points and fixed point (fire, water, road).
        #points: should be a DataFrame 
        #fp: should be a dict of a single of the fixed points e.g. {fire: {'x':1, 'y':2}}
        return np.sqrt((points.x - fp['x'])**2 + (points.y - fp['y'])**2)

    def partial(self, cohort_d, points, fp):
        #Finds the partial derivative of the part of the cost function relating to the fixed point fp
        #cohort_d: Series containing the true distances of the points to the true fixed point
        #points:  DataFrame of the hypothesized points
        #fp:  dict of a single of the hypothesized fixed points e.g. {fire: {'x':1, 'y':2}}
        distances = self.dist(points, fp) 
        differences = distances - cohort_d.values
        main_partial = (differences/distances)
        partial_x = main_partial*2*(points.x - fp['x'])
        partial_y = main_partial*2*(points.y - fp['y'])
        return partial_x, partial_y
        
    def cost_deriv(self, cohort, points, fixed_points):
        #Returns the partial derivatives of the cost function relative to the hypothesized x, y and fixed points
        #cohort:  DataFrame of the true distances
        #points:  DataFrame of the hypothesized x,y coordinates
        #fixed_points:  dict of the fixed points (fire, water, road)
        fixed = {}
        f_p_x, f_p_y = self.partial(cohort.Horizontal_Distance_To_Fire_Points, points, fixed_points['fire'])
        w_p_x, w_p_y = self.partial(cohort.Horizontal_Distance_To_Hydrology, points, fixed_points['water'])
        r_p_x, r_p_y = self.partial(cohort.Horizontal_Distance_To_Roadways, points, fixed_points['road'])
        a = 1.0/(2*len(cohort))
        fixed['fire'] = {'x': -a*f_p_x.sum(), 'y': -a*f_p_y.sum()}
        fixed['water'] = {'x': -a*w_p_x.sum(), 'y': -a*w_p_y.sum()}
        fixed['road'] = {'x': -a*r_p_x.sum(), 'y': -a*r_p_y.sum()}
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
        fixed_points['fire']['x'] = fixed_points['fire']['x'] - alpha*fp_update['fire']['x']
        fixed_points['fire']['y'] = fixed_points['fire']['y'] - alpha*fp_update['fire']['y']
        fixed_points['water']['x'] = fixed_points['water']['x'] - alpha*fp_update['water']['x']
        fixed_points['water']['y'] = fixed_points['water']['y'] - alpha*fp_update['water']['y']
        fixed_points['road']['x'] = fixed_points['road']['x'] - alpha*fp_update['road']['x']
        fixed_points['road']['y'] = fixed_points['road']['y'] - alpha*fp_update['road']['y']
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
        threshold = .5
        #print p.loc[4]
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
                if periodic_costs[-1] < threshold:
                    print "Breaking off since cost ({:.3f}) is less than threshold value ({})".format(periodic_costs[-1], threshold)
                    break
        #print p.loc[4]
        return p, fp, periodic_costs

    def automate_jiggle(self, data, p, fp):
        order = self.examine_results(data, p, fp)
        indices = order.sort('total', ascending=False).head(5).index
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
            cx = -fixed_points[about]['x']
            cy = -fixed_points[about]['y']
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
        cost_threshold = .5  #Cost at which we call it good enough
        p, fp, costs = self.iterate_hypothesis(n, data, alpha, None, None, show_costs)
        loop_count = 1
        adjust_count = 0
        cost_change = (costs[-2] - costs[-1])/costs[-1]
        worst_index = -1
        worst_change_count = 0
        best_jiggle = [0,0,0]
        
        while costs[-1] > cost_threshold:
            while cost_change is None or cost_change >= inc_improvement_threshold:
                if cost_change is not None:
                    print "Doing well (change of {:.3f}).  Cost {:.3f}.  Keeping at it.".format(cost_change, costs[-1])
                loop_count += 1
                p, fp, costs = self.iterate_hypothesis(n, data, alpha, p, fp, show_costs)
                if costs[-1] <= 1:  #At this point absolute change more compelling than relative change
                    cost_change = (costs[-2] - costs[-1])
                else:
                    cost_change = (costs[-2] - costs[-1])/costs[-1]
                if costs[-1] < cost_threshold:
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
            fixed_points['fire']['x'] += x_amount
            fixed_points['fire']['y'] += y_amount
            fixed_points['water']['x'] += x_amount
            fixed_points['water']['y'] += y_amount
            fixed_points['road']['x'] += x_amount
            fixed_points['road']['y'] += y_amount
        
    def points_fp_to_vector(self, points, fp):
        return np.concatenate([self.fp_to_vector(fp), self.points_to_vector(points)])

    def points_to_vector(self, points):
        return np.concatenate([points.x.values, points.y.values])

    def points_from_vector(self, x):
        m = len(x)/2
        points = pd.DataFrame(np.zeros((m, 2)), columns=['x', 'y'])
        points.x = x[0:m]
        points.y = x[m:]
        return points


    def fp_to_array(self, fp):
        A = np.zeros((2, 3))
        A[0,0] = fp['fire']['x']
        A[1,0] = fp['fire']['y']
        A[0,1] = fp['water']['x']
        A[1,1] = fp['water']['y']
        A[0,2] = fp['road']['x']
        A[1,2] = fp['road']['y']
        return A

    def fp_to_vector(self, fp):
        x = np.zeros(6)
        x[0] = fp['fire']['x']
        x[1] = fp['fire']['y']
        x[2] = fp['water']['x']
        x[3] = fp['water']['y']
        x[4] = fp['road']['x']
        x[5] = fp['road']['y']
        return x

    def fp_from_vector(self, x):
        fp = {'fire': {'x':0, 'y':0}, 'water':{'x':0, 'y':0}, 'road':{'x':0, 'y':0}}
        fp['fire']['x'] = x[0]
        fp['fire']['y'] = x[1]
        fp['water']['x'] = x[2]
        fp['water']['y'] = x[3]
        fp['road']['x'] = x[4]
        fp['road']['y'] = x[5]
        return fp
        

    def fp_from_array(self, fp, A):
        fp['fire']['x'] = A[0, 0]
        fp['fire']['y'] = A[1,0]
        fp['water']['x'] = A[0,1]
        fp['water']['y'] = A[1,1]
        fp['road']['x'] = A[0,2]
        fp['road']['y'] = A[1,2]
        
    def rotate(self, points, fixed_points, angle):
        A = np.zeros((2,2))
        A[0,0] = np.cos(angle)
        A[1,0] = np.sin(angle)
        A[0,1] = -np.sin(angle)
        A[1,1] = np.cos(angle)
        
        return np.transpose(np.dot(A, np.transpose(points[['x', 'y']].values))), np.dot(A, fp_to_array(fixed_points))

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
        recenter(secondary_fp, secondary, diff_x, diff_y)
        
        #Next need rotation/reflection that best fits remaining overlapped points
        #We want the rotation to be around the point of commonality, so easiest if we
        #  shift that value to 0, 0, do the rotation, then shift back
        shift_x = primary.loc[overlap[0]].x
        shift_y = primary.loc[overlap[0]].y
        recenter(None, primary, -shift_x, -shift_y)
        recenter(secondary_fp, secondary, -shift_x, -shift_y)
        
        #Normal equation: Ax = y ->  Ax'x = x'y ->  A = (x'x)^-1 x'y
        X = secondary.loc[overlap, ['x', 'y']].values
        Y = primary.loc[overlap, ['x', 'y']].values
        transform = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), Y))
        secondary[['x', 'y']] = np.dot(transform, secondary[['x', 'y']].values.transpose()).transpose()
        #Need to rotate fixed points as well
        fp_array = np.dot(transform, fp_to_array(secondary_fp))
        fp_from_array(secondary_fp, fp_array)
        recenter(None, primary, shift_x, shift_y)
        recenter(secondary_fp, secondary, shift_x, shift_y)
        return True
    
    def examine_results(self, cohort, p, fp):
        #cohort is original data set, p is hypothesized points, fp is hypothesized fire, water, road points
        #prints out the difference between the hypothesized distances and true distances 
        #  between each point and the fixed points rounded to integer
        ordering = pd.DataFrame(np.zeros((len(p), 4)), columns=['fire', 'water', 'road', 'total'])
        for i in range(len(p)):
            fire_d = np.sqrt((p.x.iloc[i] - fp['fire']['x'])**2 + (p.y.iloc[i] - fp['fire']['y'])**2)
            water_d = np.sqrt((p.x.iloc[i] - fp['water']['x'])**2 + (p.y.iloc[i] - fp['water']['y'])**2)
            road_d = np.sqrt((p.x.iloc[i] - fp['road']['x'])**2 + (p.y.iloc[i] - fp['road']['y'])**2)
            true_f = cohort.Horizontal_Distance_To_Fire_Points.iloc[i]
            true_w = cohort.Horizontal_Distance_To_Hydrology.iloc[i]
            true_r = cohort.Horizontal_Distance_To_Roadways.iloc[i]
            ordering.loc[i, ['fire', 'water', 'road']] = [fire_d - true_f, water_d - true_w, road_d - true_r]
        ordering['total'] = abs(ordering['fire']) + abs(ordering['water']) + abs(ordering['road'])
        return ordering


    
# Functions to create test data and display results




def plot_results(true_points, hyp_points, true_fixed_points, hyp_fixed_points, inc_true=True, inc_hyp=True):
    #Plot the hypothesized points and fixed points as well as the true points and fixed points.
    #  The true values will plot as circles, the hypothesized as x.
    #  The fixed points will be larger with same shape scheme and red for fire, blue for water, black for road
    if inc_true:
        plt.scatter(true_points.x, true_points.y, c=range(0, len(hyp_points)), marker='o', s=60)
        plt.scatter([true_fixed_points['fire']['x']], [true_fixed_points['fire']['y']], c='red', s=250)
        plt.scatter([true_fixed_points['water']['x']], [true_fixed_points['water']['y']], c='blue', s=250)
        plt.scatter([true_fixed_points['road']['x']], [true_fixed_points['road']['y']], c='black', s=250)
    if inc_hyp:
        plt.scatter(hyp_points.x, hyp_points.y, c=range(0, len(hyp_points)), marker='x', s=60)
        plt.scatter([hyp_fixed_points['fire']['x']], [hyp_fixed_points['fire']['y']], c='red', marker='x', s=250)
        plt.scatter([hyp_fixed_points['water']['x']], [hyp_fixed_points['water']['y']], c='blue', marker='x', s=250)
        plt.scatter([hyp_fixed_points['road']['x']], [hyp_fixed_points['road']['y']], c='black', marker='x', s=250)

    plt.show()




def compare_plots(true_p, hyp_p, true_fp, hyp_fp, rotation, reflection=None):
    plt.figure(figsize=(12, 8))
    pr, fpr = rotate(hyp_p, hyp_fp, rotation)
#    maxval = max([max(abs(fpr)), max(abs(pr))
    plt.axis('equal')
    # negative x-values to make a reflection over the y-axis
    if reflection is not None and reflection.lower() == 'y':
        pr[:,0] = -pr[:,0]
        fpr[0,:] = -fpr[0,:]
    elif reflection is not None and reflection.lower() == 'x':
        pr[:,1] = -pr[:,1]
        fpr[1,:] = -fpr[1,:]
    plt.scatter(pr[:, 0], pr[:, 1], c=range(0, len(pr[:,0])), marker='x', s=60)
    plt.scatter(fpr[0,0], fpr[1, 0], c='red', marker='x', s=250)
    plt.scatter(fpr[0,1], fpr[1, 1], c='blue', marker='x', s=250)
    plt.scatter(fpr[0,2], fpr[1, 2], c='black', marker='x', s=250)
    plot_results(true_p, hyp_p, true_fp, hyp_fp, inc_hyp=False)

#plt.scatter(df.Horizontal_Distance_To_Hydrology, 
#            df.Horizontal_Distance_To_Roadways, c='blue')
#plt.scatter(df.loc[cohorts[i].index].Horizontal_Distance_To_Hydrology, 
#            df.loc[cohorts[i].index].Horizontal_Distance_To_Roadways, c='red')

#plt.scatter(df.loc[t1].Horizontal_Distance_To_Hydrology, df.loc[t1].Horizontal_Distance_To_Roadways, s=200, c='red')
