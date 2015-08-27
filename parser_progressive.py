"""
 For the moment I am leaving this parser alone in favor of the fpblanket parser
which seems to have a number of advantages over this one.  It might be that as 
I start working more with the real data that this one starts to look better though;
we will see.
"""

import numpy as np
import pandas as pd
import scipy
import scipy.optimize as opt
from geographic_test import GeoParser

import time


class ProgressiveParser(GeoParser):
    """ GeoParser which first attempts to find a good fitting neighborhood of points and fixed points,
    then tries to extend from that neighborhood accumulating progressively more points in a good cohort.
    """

    def __init__(self, data, radius=500):
        super(ProgressiveParser, self).__init__(data, radius)
        self.init_fixed_points()
    
    def init_fixed_points(self):
        self.df_fixed_points = pd.DataFrame(columns=['x', 'y', 'type'])
    
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

    def add_fixed_points(self, fp_set):
        """Compares to fixed points already in self.df_fixed_points.  If close to a point already there, treats it as
        same point and doesn't add it.  Otherwise appends the fixed point to self.df_fixed_points.
        :fp_set should be DataFrame of fire, water, road fixed points

        """
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
        """Attempts to match up self.current_cohort to self.accumulated_cohorts.  Needs at least 3 overlapping points to 
        be matched up theoretically the overlap could include the fixed points, but in practice that might be a little
        tricky since additional checking would need to be done to ensure the fixed points were the same.
        If they can be matched, appends self.current_cohort to self.accumulated_cohorts using the accumulated_cohorts values
        where there is overlap.  Also calls self.add_fixed_points(self.fixed_points).
        """
        joined_points = self.accumulated_cohorts
        if len(set(joined_points.index) & set(self.current_cohort.index)) >= 3:
            self.align_cohorts(joined_points, self.current_cohort, self.fixed_points)
            joined_points = pd.concat([joined_points, 
                self.current_cohort.loc[set(self.current_cohort.index) - set(joined_points.index)]])
            self.add_fixed_points(self.fixed_points)
            self.accumulated_cohorts = joined_points
        else:
            print "Not enough overlap between accumulated cohorts and current cohort?!" #Should not happen


