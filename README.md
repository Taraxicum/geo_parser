# Inverse Geographic Mapping project 
(See http://www.datanaturally.com/2015/05/inverse-geographic-mapping-introduction.html for a description of the problem.)


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
