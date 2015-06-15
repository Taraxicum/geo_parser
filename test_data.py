import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        """self.points and self.fixed_points need to be defined already.
        Will create a DataFrame containing the distances from the points to the fixed points
        This is intended to build a DataFrame with values similar to what we get from the kaggle competition from
          the points and fixed_point coordinates that we generated for testing
        self.points should be a DataFrame of x, y coordinates
        self.fixed_points should be a dict of fixed point coordinates (fire, water, road)
        """
        self.data = pd.DataFrame(np.zeros((len(self.points), 3)), columns=['Horizontal_Distance_To_Fire_Points',
                                                               'Horizontal_Distance_To_Hydrology',
                                                               'Horizontal_Distance_To_Roadways'])
        self.data['Horizontal_Distance_To_Fire_Points'] = self.distance_to_fp('fire')
        self.data['Horizontal_Distance_To_Hydrology'] = self.distance_to_fp('water')
        self.data['Horizontal_Distance_To_Roadways'] =  self.distance_to_fp('road')

    def distance_to_fp(self, fp_type):
        """
        :fp_type: should be one of fire, water, or road
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
