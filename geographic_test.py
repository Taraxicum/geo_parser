import numpy as np
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
        self.fps = []
        self.make_cohorts(1500)
        self.automated_iteration(self.cohorts[0], 1000, 1, False)

    
    #Make/filter cohorts -> map cohorts to coordinates -> match coordinates up for overlapping cohorts -> done?


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
        remaining = self.data.index
        
        while len(remaining) > 0:
            p = remaining[np.random.randint(len(remaining))]
            self.cohorts.append(self.find_cohort(p, radius))
            remaining = remaining - self.cohorts[-1].index
        print len(self.cohorts)
        for i, c in enumerate(self.cohorts):
            if len(c) < self.cohort_threshold:
                del self.cohorts[i]  
            else:
                self.cohorts[i] = self.data.loc[c.index]
                print "Cohort {}, length {}".format(i, len(c))

    def cost(self, cohort, points, fixed_points):
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
                print "Iteration {}: cost {:.3f}, digit count {}".format(i+1, periodic_costs[-1], digits)
                if periodic_costs[-1] < threshold:
                    print "Breaking off since cost ({:.3f}) is less than threshold value ({})".format(periodic_costs[-1], threshold)
                    break
        #print p.loc[4]
        return p, fp, periodic_costs

    def jiggle(self, indices, points, fixed_points, about='fire', axis='both'):
        for index in indices:
            self.recenter(fixed_points, points, -fixed_points[about]['x'], -fixed_points[about]['y'])
            if axis == 'x' or axis == 'both':
                points.loc[index, 'x'] = - points.loc[index, 'x']
            elif axis == 'y' or axis == 'both':
                points.loc[index, 'y'] = - points.loc[index, 'y']


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
        


    def fp_to_array(self, fp):
        A = np.zeros((2, 3))
        A[0,0] = fp['fire']['x']
        A[1,0] = fp['fire']['y']
        A[0,1] = fp['water']['x']
        A[1,1] = fp['water']['y']
        A[0,2] = fp['road']['x']
        A[1,2] = fp['road']['y']
        return A

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
