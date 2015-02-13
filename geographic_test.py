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
        self.data = pd.DataFrame(np.zeros((len(points), 3)), columns=['Horizontal_Distance_To_Fire_Points',
                                                               'Horizontal_Distance_To_Hydrology',
                                                               'Horizontal_Distance_To_Roadways'])
        self.data['Horizontal_Distance_To_Fire_Points'] = np.sqrt((self.points.x - self.fixed_points['fire']['x'])**2 + 
                                                           (self.points.y - self.fixed_points['fire']['y'])**2)
        self.data['Horizontal_Distance_To_Hydrology'] = np.sqrt((self.points.x - self.fixed_points['water']['x'])**2 + 
                                                         (self.points.y - self.fixed_points['water']['y'])**2)
        self.data['Horizontal_Distance_To_Roadways'] = np.sqrt((self.points.x - self.fixed_points['road']['x'])**2 + 
                                                        (self.points.y - self.fixed_points['road']['y'])**2)


class GeoParser():
    def __init__(self):

def norm_distances(cohort):
    norm_param = pd.DataFrame(np.zeros((2, 3)), index=['mu', 'sigma'], columns=['fire', 'water', 'road'])
    norm_param.loc['mu', 'fire'] = cohort.Horizontal_Distance_To_Fire_Points.mean()
    norm_param.loc['sigma', 'fire'] = cohort.Horizontal_Distance_To_Fire_Points.std()
    norm_param.loc['mu', 'water'] = cohort.Horizontal_Distance_To_Hydrology.mean()
    norm_param.loc['sigma', 'water'] = cohort.Horizontal_Distance_To_Hydrology.std()
    norm_param.loc['mu', 'road'] = cohort.Horizontal_Distance_To_Roadways.mean()
    norm_param.loc['sigma', 'road'] = cohort.Horizontal_Distance_To_Roadways.std()
    cohort['fire_normed'] = (cohort.Horizontal_Distance_To_Fire_Points - norm_param.fire.mu)/norm_param.fire.sigma
    cohort['water_normed'] = (cohort.Horizontal_Distance_To_Hydrology - norm_param.water.mu)/norm_param.water.sigma
    cohort['road_normed'] = (cohort.Horizontal_Distance_To_Roadways - norm_param.road.mu)/norm_param.road.sigma
    return norm_param

def cost(cohort, points, fixed_points, norm_param=None):
    #cohort of points we are trying to map to a 2d representation that fits the data
    #points are the x, y coordinates of the hypothesized points (should be len(cohort) of them)
    #fixed_points x, y coordinates of the hypothesized fire, water, road locations
    #This is not the exact same cost function the derivative of the cost function uses - this 
    #  one uses square root to make the values a little easier to think of as an average error
    #  but would unnecessarily complicate the derivative
    if norm_param is None:
        fire_d = np.sqrt((dist(points, fixed_points['fire']) - cohort.Horizontal_Distance_To_Fire_Points.values)**2)
        water_d = np.sqrt((dist(points, fixed_points['water']) - cohort.Horizontal_Distance_To_Hydrology.values)**2)
        road_d = np.sqrt((dist(points, fixed_points['road']) - cohort.Horizontal_Distance_To_Roadways.values)**2)
    else:
        fire_d = np.sqrt((dist(points, fixed_points['fire'], norm_param.fire) - cohort.fire_normed.values)**2)
        water_d = np.sqrt((dist(points, fixed_points['water'], norm_param.water) - cohort.water_normed.values)**2)
        road_d = np.sqrt((dist(points, fixed_points['road'], norm_param.road) - cohort.road_normed.values)**2)
    return 1.0/(2*len(cohort))*(fire_d.sum() + water_d.sum() + road_d.sum())

def dist(points, fp, norm_param=None):
    #Distance between points and fixed point (fire, water, road).
    #points: should be a DataFrame 
    #fp: should be a dict of a single of the fixed points e.g. {fire: {'x':1, 'y':2}}
    if norm_param is None:
        return np.sqrt((points.x - fp['x'])**2 + (points.y - fp['y'])**2)
    else:
        return (np.sqrt((points.x - fp['x'])**2 + (points.y - fp['y'])**2) - norm_param.mu)/norm_param.sigma
    

def partial(cohort_d, points, fp, norm_param=None):
    #Finds the partial derivative of the part of the cost function relating to the fixed point fp
    #cohort_d: Series containing the true distances of the points to the true fixed point
    #  or the normed true distances if norm_param is not None
    #points:  DataFrame of the hypothesized points
    #fp:  dict of a single of the hypothesized fixed points e.g. {fire: {'x':1, 'y':2}}
    #norm_param: mean and standard deviation used to normalize distances relative to fp
    
    distances = dist(points, fp) #In this case we want the distances, not the normed distances regarless of 
                                 #  whether we are otherwise normalizing   
    if norm_param is None:
        differences = distances - cohort_d.values
        main_partial = (differences/distances)
    else:
        differences = (distances - norm_param.mu)/norm_param.sigma - cohort_d.values
        main_partial = differences/(distances*norm_param.sigma)
    partial_x = main_partial*2*(points.x - fp['x'])
    partial_y = main_partial*2*(points.y - fp['y'])
    return partial_x, partial_y
    
def cost_deriv(cohort, points, fixed_points, norm_param=None):
    #Returns the partial derivatives of the cost function relative to the hypothesized x, y and fixed points
    #cohort:  DataFrame of the true distances
    #points:  DataFrame of the hypothesized x,y coordinates
    #fixed_points:  dict of the fixed points (fire, water, road)
    #norm_param:  parameters used to normalize features if normalized
    fixed = {}
    if norm_param is None:
        f_p_x, f_p_y = partial(cohort.Horizontal_Distance_To_Fire_Points, points, fixed_points['fire'])
        w_p_x, w_p_y = partial(cohort.Horizontal_Distance_To_Hydrology, points, fixed_points['water'])
        r_p_x, r_p_y = partial(cohort.Horizontal_Distance_To_Roadways, points, fixed_points['road'])
    else:
        f_p_x, f_p_y = partial(cohort.fire_normed, points, fixed_points['fire'], norm_param.fire)
        w_p_x, w_p_y = partial(cohort.water_normed, points, fixed_points['water'], norm_param.water)
        r_p_x, r_p_y = partial(cohort.road_normed, points, fixed_points['road'], norm_param.road)
    a = 1.0/(2*len(cohort))
    fixed['fire'] = {'x': -a*f_p_x.sum(), 'y': -a*f_p_y.sum()}
    fixed['water'] = {'x': -a*w_p_x.sum(), 'y': -a*w_p_y.sum()}
    fixed['road'] = {'x': -a*r_p_x.sum(), 'y': -a*r_p_y.sum()}
    partial_x = a*(f_p_x + w_p_x + r_p_x)
    partial_y = a*(f_p_y + w_p_y + r_p_y)
    return partial_x, partial_y, fixed
    

def init_points(cohort):
    #Initializes set of hypothesized points and fixed points for iterating our gradient descent algorithm
    #cohort:  DataFrame containing the true distances of the points to the fixed points
    points = pd.DataFrame(100*np.random.randn(len(cohort), 2), columns=['x', 'y'])
    vals = 1000*np.random.randn(6)
    fixed_points = {'fire': {'x': vals[0], 'y':vals[1]}, 'water': {'x': vals[2], 'y':vals[3]}, 
                    'road': {'x': vals[4], 'y':vals[5]}}
    return points, fixed_points

def update_values(points, fixed_points, alpha, x_update, y_update, fp_update):
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

def iterate_hypothesis(n, cohort, alpha=.02, p=None, fp=None, norm_param=None, show_costs=True):
    #Iterates the gradient descent algorithm
    #Prints out cost of the hypothesized coordinates at intervals throughout the iteration
    #Returns p, fp that have resulted at end of n iterations
    #n:  Number of iterations (integer)
    #cohort:  DataFrame containing true distances of points from fixed points
    #alpha:  learning rate (positive real number
    #p:  DataFrame of hypothesized x, y coordinates.  Will initialize to random values if not supplied
    #fp: DataFrame of hypothesized fixed point coordinates.  Will initialize to random values if not supplied
    periodic_costs = []
    if norm_param is None:
        threshold = .5
    else:
        threshold = .003
    if p is None or fp is None:
        print "initializing points"
        p, fp = init_points(cohort)
    periodic_costs.append(cost(cohort, p, fp, norm_param))
    if norm_param is None and show_costs:
        print "Initial cost {:.3f}".format(periodic_costs[-1])
    elif show_costs:
        print "Initial cost {:.3f}".format(periodic_costs[-1])
    digit_size = int(np.log10(n))
    print_mod = 10**(digit_size-1)
    for i in range(n):
        px, py, pfix = cost_deriv(cohort, p, fp, norm_param)
        p, fp = update_values(p, fp, alpha, px, py, pfix)
        
        if (i+1)%print_mod == 0:
            periodic_costs.append(cost(cohort, p, fp, norm_param))
            digits = int(np.log10(periodic_costs[-1] + 1) + 1)
            if norm_param is None and show_costs:
                print "Iteration {}: cost {:.3f}, digit count {}".format(i+1, periodic_costs[-1], digits)
            elif show_costs:
                print "Iteration {}: cost {:.3f}, digit count {}".format(i+1, periodic_costs[-1], digits)
            if periodic_costs[-1] < threshold:
                print "Breaking off since cost ({:.3f}) is less than threshold value ({})".format(periodic_costs[-1], threshold)
                break
    return p, fp, periodic_costs


# Functions to create test data and display results



def examine_results(cohort, p, fp):
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


def jiggle(indices, points, fixed_points, about='fire', axis='both'):
    for index in indices:
        recenter(fixed_points, points, -fixed_points[about]['x'], -fixed_points[about]['y'])
        if axis == 'x' or axis == 'both':
            points.loc[index, 'x'] = - points.loc[index, 'x']
        elif axis == 'y' or axis == 'both':
            points.loc[index, 'y'] = - points.loc[index, 'y']


def automated_iteration(data, n=1000, alpha=2, show_costs=False):
    inc_improvement_threshold = .008 #Relative change in cost at which we wnt to adjust values before continuing
                                    #Probably makes sense for this to be different depending on n
                                    #  but if we are reasonably consistent with n can just hand tune as needed
    cost_threshold = .5  #Cost at which we call it good enough
    p, fp, costs = iterate_hypothesis(n, data, alpha, None, None, None, show_costs)
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
            p, fp, costs = iterate_hypothesis(n, data, alpha, p, fp, None, show_costs)
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
        order = examine_results(data, p, fp)
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
            jiggle(indices, p, fp, about, 'y')
        elif worst_change_count == 1:
            jiggle(indices, p, fp, about, 'x')
        elif worst_change_count == 2:
            jiggle(indices, p, fp, about, 'both')
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

#This was the second pass through iterating 10000 times.  In between the first and second pass I hand
#  adjusted one of the hypothesized points that seemed to be stuck in a local minima and the set quickly converged
#  after that.
#p, fp, costs = iterate_hypothesis(1000, df, 2)#, p, fp, None)#, norm_p)
#%timeit -n 10 -r 1 iterate_hypothesis(1000, df, 1.5, None, None, None, False)
#print costs
#p, fp, costs = automated_iteration(df, n=1000)


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
        fixed_points['fire']['x'] += x_amount
        fixed_points['fire']['y'] += y_amount
        fixed_points['water']['x'] += x_amount
        fixed_points['water']['y'] += y_amount
        fixed_points['road']['x'] += x_amount
        fixed_points['road']['y'] += y_amount
    


def fp_to_array(fp):
    A = np.zeros((2, 3))
    A[0,0] = fp['fire']['x']
    A[1,0] = fp['fire']['y']
    A[0,1] = fp['water']['x']
    A[1,1] = fp['water']['y']
    A[0,2] = fp['road']['x']
    A[1,2] = fp['road']['y']
    return A

def fp_from_array(fp, A):
    fp['fire']['x'] = A[0, 0]
    fp['fire']['y'] = A[1,0]
    fp['water']['x'] = A[0,1]
    fp['water']['y'] = A[1,1]
    fp['road']['x'] = A[0,2]
    fp['road']['y'] = A[1,2]
    
def rotate(points, fixed_points, angle):
    A = np.zeros((2,2))
    A[0,0] = np.cos(angle)
    A[1,0] = np.sin(angle)
    A[0,1] = -np.sin(angle)
    A[1,1] = np.cos(angle)
    
    return np.transpose(np.dot(A, np.transpose(points[['x', 'y']].values))), np.dot(A, fp_to_array(fixed_points))


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

# <headingcell level=3>

# Run the algorithm


def distance_3d(p1, p2):
    return np.sqrt((p1['Horizontal_Distance_To_Fire_Points'] - p2['Horizontal_Distance_To_Fire_Points'])**2 +
                   (p1['Horizontal_Distance_To_Hydrology'] - p2['Horizontal_Distance_To_Hydrology'])**2 +
                   (p1['Horizontal_Distance_To_Roadways'] - p2['Horizontal_Distance_To_Roadways'])**2)

def find_cohort(pix, radius, df): 
    ds = pd.DataFrame(np.zeros((len(df), 1)), columns=['distance'])
    for i in range(len(df)):
        ds.loc[i, 'distance'] = distance_3d(df.loc[pix], df.loc[i])
    return ds.loc[ds.distance < radius]

def make_cohorts(radius, df):
    #From testing, it appears that having cohorts of at least size 4 is desirable, I did have one example
    #  at size 4 that failed to converge nicely, but several other examples worked quite well.
    #Size 3 had some tests that worked great and some that did converged to a non-true solution
    cohorts = []
    remaining = df.index
    
    while len(remaining) > 0:
        p = remaining[np.random.randint(len(remaining))]
        cohorts.append(find_cohort(p, radius, df))
        remaining = remaining - cohorts[-1].index
    return cohorts


#Creates a set of n 'true' coordinates for a test set of points and fixed_points then generate the DataFrame
#  containing the distances of the points to the fixed points


fps = []
cohorts = make_cohorts(1500, df)
print len(cohorts)
for i in range(len(cohorts)):
    cohorts[i] = df.loc[cohorts[i].index]
    print "Cohort {}, length {}".format(i, len(cohorts[i]))
#plt.scatter(df.Horizontal_Distance_To_Hydrology, 
#            df.Horizontal_Distance_To_Roadways, c='blue')
#plt.scatter(df.loc[cohorts[i].index].Horizontal_Distance_To_Hydrology, 
#            df.loc[cohorts[i].index].Horizontal_Distance_To_Roadways, c='red')

#plt.scatter(df.loc[t1].Horizontal_Distance_To_Hydrology, df.loc[t1].Horizontal_Distance_To_Roadways, s=200, c='red')


def align_cohorts(primary, secondary, secondary_fp):
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
    
    


c0, c1, fp1 = align_cohorts(cohorts[0], cohorts[3], fps[-1])


print cohorts[0].tail()
print cohorts[3].tail()
print fps[0]
print fps[-1]


#print cohorts[0].head()
#print cohorts[1].head()
#print fps[0]
#print fps[1]
tojoin = cohorts[3].index - cohorts[0].index
joined = pd.concat([cohorts[0], cohorts[3].loc[tojoin]])
print len(joined)
print joined.head()


cohort = cohorts[3]
print len(cohort)
p, fp, costs = automated_iteration(cohort, n=500, alpha=1)


cohort['x'] = p.x.values
cohort['y'] = p.y.values
fps.append(fp)


#Recenter for ease of comparing plots
recenter(fixed_points, points, -fixed_points['fire']['x'], -fixed_points['fire']['y'])
recenter(fp, p, -fp['fire']['x'], -fp['fire']['y'])


compare_plots(points, cohorts[0], fixed_points, fps[0], 1.2*np.pi/6 + np.pi, "")
compare_plots(points, joined[['x', 'y']], fixed_points, fps[-1], 1.2*np.pi/6 + np.pi, "")


print fp
print np.sqrt((fp['fire']['x'] - fp['water']['x'])**2 + (fp['fire']['y'] - fp['water']['y'])**2)
print np.sqrt((fp['fire']['x'] - fp['road']['x'])**2 + (fp['fire']['y'] - fp['road']['y'])**2)
print np.sqrt((fp['road']['x'] - fp['water']['x'])**2 + (fp['road']['y'] - fp['water']['y'])**2)

