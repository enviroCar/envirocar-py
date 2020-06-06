from scipy import interpolate
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import random
import string


class Preprocessing():
    def __init__(self):
        print("Initializing pre-processing class")   # do we need anything?

    def remove_outliers(self, points, column):
        """ Remove outliers by using the statistical approach
        as described in
        https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm

        Keyword Arguments:
            points {GeoDataFrame} -- A GeoDataFrame containing the track points
            column {String} -- Columnn name to remove outliers from

        Returns:
            new_points -- Points with outliers removed
        """

        # print(points['time'])
        first_quartile = points[column].quantile(0.25)
        third_quartile = points[column].quantile(0.75)
        iqr = third_quartile-first_quartile   # Interquartile range
        fence_low = first_quartile - 1.5 * iqr
        fence_high = third_quartile + 1.5 * iqr

        new_points = points.loc[(points[column] > fence_low) & (
            points[column] < fence_high)]
        # print(new_points['time'])
        return new_points

    def interpolate(self, points):
        """ Creates a trajectory from point data

        Keyword Arguments:
            points {GeoDataFrame} -- A GeoDataFrame containing the track points

        Returns:
            new_points -- An interpolated trajectory
        """

        def date_to_seconds(x):
            date_time_obj = datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')
            seconds = (date_time_obj-datetime.datetime(1970, 1, 1)
                       ).total_seconds()
            return int(seconds)

        def seconds_to_date(x):
            date = datetime.datetime.fromtimestamp(x, datetime.timezone.utc)
            return date

        def randStr(chars=string.ascii_uppercase + string.digits, N=24):
            return ''.join(random.choice(chars) for _ in range(N))

        print(points.shape)

        # to have flat attributes for coordinates
        points['lat'] = points['geometry'].apply(lambda coord: coord.y)
        points['lng'] = points['geometry'].apply(lambda coord: coord.x)
        points_df = pd.DataFrame(points)

        # removing duplicates because interpolation won't work otherwise
        points_df_cleaned = points_df.drop_duplicates(
            ['lat', 'lng'], keep='last')

        # input for datetime in seconds
        points_df_cleaned['time_seconds'] = np.vectorize(date_to_seconds)(
            np.array(points_df_cleaned.time.values.tolist()))

        # creating the column name lists
        names_interpolate = [s for s in points_df_cleaned.columns if
                             '.value' in s]
        # adding the other column names at front
        names_interpolate = ['lng', 'lat', 'time_seconds'] + names_interpolate
        names_replicatate = np.setdiff1d(points_df_cleaned.columns,
                                         names_interpolate)
        names_extra = ['geometry', 'id', 'time']
        names_replicatate = [x for x in names_replicatate if x
                             not in names_extra]

        # measurements themselves
        columns_interpolate = [np.array(
            points_df_cleaned[column].values.tolist()) for column
            in names_interpolate]

        # split dataframe because splprep cannot take more than 11
        dfs = np.split(columns_interpolate, [4, 14], axis=0)

        """ Interpolation itself """
        # Find the B-spline representation of the curve
        # tck (t,c,k): is a tuple containing the vector of knots,
        # the B-spline coefficients, and the degree of the spline.
        # u: is an array of the values of the parameter.

        # print(type(dfs[0][2]))
        # interpolating so many points to have a point for each second
        step = np.linspace(0, 1, points_df_cleaned['time_seconds'].iloc[-1] -
                           points_df_cleaned['time_seconds'].iloc[0])
        tck_0, u_0 = interpolate.splprep(dfs[0], s=0)
        new_points_0 = interpolate.splev(step, tck_0)
        tck_1, u_1 = interpolate.splprep(dfs[1], s=0)
        new_points_1 = interpolate.splev(step, tck_1)
        tck_2, u_2 = interpolate.splprep(dfs[2], s=0)
        new_points_2 = interpolate.splev(step, tck_2)

        new_points = new_points_0 + new_points_1 + new_points_2

        # transposing the resulting matrix to fit it in the dataframe
        data = np.transpose(new_points)

        # constructing the new dataframe
        interpolated_df = pd.DataFrame(data)

        interpolated_df.columns = names_interpolate
        interpolated_df['time'] = np.vectorize(
            seconds_to_date)(interpolated_df['time_seconds'])

        # these should all be the same for one ride, so just replicating
        columns_replicate = [np.repeat(points_df_cleaned[column].iloc[0],
                             len(step)) for column in names_replicatate]

        replicated_transposed = np.transpose(columns_replicate)
        replicated_df = pd.DataFrame(replicated_transposed)
        replicated_df.columns = names_replicatate

        # combining replicated with interpolated
        full_df = pd.concat([interpolated_df, replicated_df], axis=1,
                            sort=False)

        # adding ids
        full_df['id'] = 0
        for row in full_df.index:
            full_df['id'][row] = randStr()

        # transforming back to a geodataframe
        full_gdf = gpd.GeoDataFrame(
         full_df, geometry=gpd.points_from_xy(full_df.lng, full_df.lat))

        # remove full_gdf['lng'], full_gdf['lat'] ?
        del full_gdf['time_seconds']

        print(full_gdf.shape)

        return full_gdf

    def aggregate(self, points_mp):

        # TODO aggregation of points here

        return 'Aggregation function was called. Substitute with result'

    def cluster(self, points_mp):

        # TODO clustering of points here

        return 'Clustering function was called. Substitute with result'
