from scipy import interpolate
from datetime import timedelta
import pandas as pd
import numpy as np
import datetime
import folium
import movingpandas as mpd
import enum

class GeneralizationType(enum.Enum):
    DouglasPeucker = 0
    MinDistance = 1
    MinTimeDelta = 2

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

        first_quartile = points[column].quantile(0.25)
        third_quartile = points[column].quantile(0.75)
        iqr = third_quartile-first_quartile   # Interquartile range
        fence_low = first_quartile - 1.5 * iqr
        fence_high = third_quartile + 1.5 * iqr

        new_points = points.loc[(points[column] > fence_low) & (
            points[column] < fence_high)]

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

        # to have flat attributes
        points['lat'] = points['geometry'].apply(lambda coord: coord.y)
        points['lng'] = points['geometry'].apply(lambda coord: coord.x)
        points_df = pd.DataFrame(points)

        # removing duplicates because interpolation won't work otherwise
        points_df_cleaned = points_df.drop_duplicates(
            ['lat', 'lng'], keep='last')

        # input for datetime in seconds
        input_time = np.vectorize(date_to_seconds)(
            np.array(points_df_cleaned.time.values.tolist()))

        # measurements TODO which else?
        input_co2 = np.array(points_df_cleaned['CO2.value'].values.tolist())
        input_speed = np.array(
            points_df_cleaned['Speed.value'].values.tolist())

        # input arrays for coordinates
        input_coords_y = np.array(points_df_cleaned.lat.values.tolist())
        input_coords_x = np.array(points_df_cleaned.lng.values.tolist())

        """ Interpolation itself """
        # Find the B-spline representation of the curve
        # tck (t,c,k): is a tuple containing the vector of knots,
        # the B-spline coefficients, and the degree of the spline.
        # u: is an array of the values of the parameter.
        tck, u = interpolate.splprep(
            [input_coords_x, input_coords_y, input_time, input_co2,
             input_speed], s=0)
        step = np.linspace(0, 1, input_time[-1] - input_time[0])
        # interpolating so many points to have a point for each second
        new_points = interpolate.splev(step, tck)

        # transposing the resulting matrix to fit it in the dataframe
        data = np.transpose(new_points)

        # constructing the new dataframe
        interpolated_df = pd.DataFrame(data)
        interpolated_df.columns = ['lng', 'lat',
                                   'time_seconds', 'CO2.value', 'Speed.value']
        interpolated_df['time'] = np.vectorize(
            seconds_to_date)(interpolated_df['time_seconds'])
        # TODO is there need for this column?
        interpolated_df.drop(['time_seconds'], axis=1)

        return interpolated_df

    # Creating trajectiors from each unique set of points in dataframe
    # Creats a Moving Pandas Trajectory Collection Object
    def trajectoryCollection(self, data_df, MIN_LENGTH):
        track_df = data_df

        # adding a time field as 't' for moving pandas indexing
        track_df['t'] = pd.to_datetime(track_df['time'], format='%Y-%m-%dT%H:%M:%S')
        track_df = track_df.set_index('t')

        # using moving pandas trajectory collection function to convert trajectory points into actual trajectories  
        traj_collection = mpd.TrajectoryCollection(track_df, 'track.id', min_length=MIN_LENGTH)
        print("Finished creating {} trajectories".format(len(traj_collection)))
        return traj_collection

    # Splitting Trajectories based on time gap between records to extract Trips
    def split_by_gap (self, TRAJ_COLLECTION, MIN_GAP):
        traj_collection = TRAJ_COLLECTION
        
        # using moving pandas function to split trajectories as 'trips'
        trips = traj_collection.split_by_observation_gap(timedelta(minutes=MIN_GAP))
        print("Extracted {} individual trips from {} continuous vehicle tracks".format(len(trips), len(traj_collection)))
        return trips

    # To Aggregate given Trips and get Aggregated Flows
    def aggregate(self, TRIPS, MAX_DISTANCE, MIN_DISTANCE, MIN_STOP_DURATION ):
        #%%time
        trips = TRIPS

        # using moving pandas function to Aggregate Trajectories
        aggregator = mpd.TrajectoryCollectionAggregator(trips, max_distance=MAX_DISTANCE, min_distance=MIN_DISTANCE, min_stop_duration=timedelta(minutes=MIN_STOP_DURATION))
        pts = aggregator.get_significant_points_gdf()
        clusters = aggregator.get_clusters_gdf()
        flows = aggregator.get_flows_gdf()
        return flows

    def cluster(self, points_mp):

        # TODO clustering of points here

        return 'Clustering function was called. Substitute this string with clustering result'

    def generalize(self, traj, tolerance, generalizationType):
        """ Generalizes the moving pandas trajectory or trajectory collection

        Keyword Arguments:
            traj -- movingpandas trajectory/trajectory collection  
            tolerance -- tolerance from 0 to 1. Specify minutes incase of MinTimDelta generalization type
            generalizationType -- type of generalization e.g. Douglas Peucker, Min Distance, or Min Time Delta

        Returns:
            moving pandas trajectory or trajectory collection
        """

        if (not isinstance(generalizationType, GeneralizationType)):
            raise ValueError("Invalid generalization type " + str(generalizationType))

        if generalizationType.value == GeneralizationType.DouglasPeucker.value:
            return mpd.DouglasPeuckerGeneralizer(traj).generalize(tolerance=tolerance)
        elif generalizationType.value == GeneralizationType.MinDistance.value:
            return mpd.MinDistanceGeneralizer(traj).generalize(tolerance=tolerance)
        elif generalizationType.value == GeneralizationType.MinTimeDelta.value:
            return mpd.MinTimeDeltaGeneralizer(traj).generalize(tolerance=timedelta(minutes=tolerance))
        