# import os
from math import floor, ceil
from scipy import interpolate
from statistics import mean
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import random
import string
from copy import copy

import folium
import movingpandas as mpd
# from shapely.geometry import Point, LineString, Polygon
from shapely.geometry import Polygon
import json
from branca.colormap import linear
# import enum


class Preprocessing():
    def __init__(self):
        print("Initializing pre-processing class")   # do we need anything?

    # Creating trajectiors from each unique set of points in dataframe
    # Creats a Moving Pandas Trajectory Collection Object
    def trajectoryCollection(self, data_df, MIN_LENGTH):
        track_df = data_df

        # adding a time field as 't' for moving pandas indexing
        track_df['t'] = pd.to_datetime(track_df['time'],
                                       format='%Y-%m-%dT%H:%M:%S')
        track_df = track_df.set_index('t')

        # using moving pandas trajectory collection function to convert
        # trajectory points into actual trajectories
        traj_collection = mpd.TrajectoryCollection(track_df, 'track.id',
                                                   min_length=MIN_LENGTH)
        print("Finished creating {} trajectories".format(len(traj_collection)))
        return traj_collection

    # Splitting Trajectories based on time gap between records to extract Trips
    def split_by_gap(self, TRAJ_COLLECTION, MIN_GAP):
        traj_collection = TRAJ_COLLECTION

        # using moving pandas function to split trajectories as 'trips'
        trips = traj_collection.split_by_observation_gap(
            datetime.timedelta(minutes=MIN_GAP))
        print("Extracted {} individual trips from {} continuous vehicle \
            tracks".format(len(trips), len(traj_collection)))
        return trips

    def calculateAcceleration(self, points_df):
        points_df['t'] = pd.to_datetime(
             points_df['time'], format='%Y-%m-%dT%H:%M:%S')

        time_arr = points_df['t'].tolist()
        speed_arr = points_df['Speed.value'].to_numpy()
        acceleration_array = [0]

        for i in range(1, len(time_arr)):
            # using speed not to calculate velocity because we don't care
            # about direction anyway
            velocity_change = speed_arr[i] - speed_arr[i-1]
            time_change = (time_arr[i] - time_arr[i-1]).total_seconds()

            if (time_change != 0):
                acceleration = (velocity_change / time_change)
            else:
                acceleration = 0

            # print(velocity_change, time_change, acceleration)
            acceleration_array.append(acceleration)

        points_df['Acceleration.value'] = acceleration_array

        return points_df

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

        def interpolate_spline(input_array, step):
            tck, u = interpolate.splprep(input_array, s=0)
            interpolated = interpolate.splev(step, tck)
            return interpolated

        def interpolate_linear(step_init, values, step_final):
            f = interpolate.interp1d(step_init, values, axis=0,
                                     fill_value="extrapolate")
            values_new = f(step_final)
            return values_new

        print('Amount of points before interpolation',
              points.shape)

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
        dfs = np.split(columns_interpolate, [2], axis=0)

        """ Interpolation itself """
        # Find the B-spline representation of the curve
        # tck (t,c,k): is a tuple containing the vector of knots,
        # the B-spline coefficients, and the degree of the spline.
        # u: is an array of the values of the parameter.

        # interpolating so many points to have a point for each second
        step = np.linspace(0, 1, points_df_cleaned['time_seconds'].iloc[-1] -
                           points_df_cleaned['time_seconds'].iloc[0])

        seconds = np.linspace(points_df_cleaned['time_seconds'].iloc[0],
                              points_df_cleaned['time_seconds'].iloc[-1],
                              points_df_cleaned['time_seconds'].iloc[-1] -
                              points_df_cleaned['time_seconds'].iloc[0])

        new_points = interpolate_spline(dfs[0], step)

        for idx, column in enumerate(dfs[1]):
            if (idx == 0):
                new_points.append(seconds)
            else:
                new_points.append(interpolate_linear(points_df_cleaned[
                 'time_seconds'], column, seconds))

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

        print('Amount of points after interpolation',
              full_gdf.shape)

        return full_gdf

    def aggregate(self, track_df, MIN_LENGTH, MIN_GAP, MAX_DISTANCE,
                  MIN_DISTANCE, MIN_STOP_DURATION):
        """ Transforms to Moving Pandas, Converts into Trajectories,
            Ignore small trajectories and return Aggregated Flows

        Keyword Arguments:
            track_df {GeoDataFrame} -- A Moving Pandas GeoDataFrame containing
                the track points
            MIN_LENGTH {integer} -- Minimum Length of a Trajectory
                (to be considered as a Trajectory)
            MIN_GAP {integer} -- Minimum Gap (in minutes) for splitting
                single Trajectory into more
            MAX_DISTANCE {integer} -- Max distance between significant points
            MIN_DISTANCE {integer} -- Min distance between significant points
            MIN_STOP_DURATION {integer} -- Minimum duration (in minutes)
                required for stop detection

        Returns:
            flows -- A GeoDataFrame containing Aggregared Flows (linestrings)
        """

        # Using MPD function to convert trajectory points into actual
        #  trajectories
        traj_collection = mpd.TrajectoryCollection(track_df, 'track.id',
                                                   min_length=MIN_LENGTH)
        print("Finished creating {} trajectories".format(len(traj_collection)))

        # Using MPD function to Split Trajectories based on time gap between
        # records to extract Trips
        trips = traj_collection.split_by_observation_gap(datetime.timedelta(
            minutes=MIN_GAP))
        print("Extracted {} individual trips from {} continuous vehicle \
            tracks".format(len(trips), len(traj_collection)))

        # Using MPD function to Aggregate Trajectories
        aggregator = mpd.TrajectoryCollectionAggregator(
                trips, max_distance=MAX_DISTANCE,
                min_distance=MIN_DISTANCE,
                min_stop_duration=datetime.timedelta(
                    minutes=MIN_STOP_DURATION))
        flows = aggregator.get_flows_gdf()
        return flows

    def aggregateByGrid(self, df, field, summary, gridSize):
        """
        Aggregates the specified field with chosen summary type and
            user defined grid size. returns aggregated grids with summary

        Parameters
        ----------
        df : geopandas dataframe
        field : string
            field to be summarized.
        summary : string
            type of summary to be sumarized. eg. min, max,sum, median
        gridSize : float
            the size of grid on same unit as geodataframe coordinates.

        Returns
        -------
        geodataframe
            Aggregated grids with summary on it

        """
        def round_down(num, divisor):
            return floor(num / divisor) * divisor

        def round_up(num, divisor):
            return ceil(num / divisor) * divisor

        xmin, ymin, xmax, ymax = df.total_bounds
        height, width = gridSize, gridSize
        top, left = round_up(ymax, height), round_down(xmin, width)
        bottom, right = round_down(ymin, height), round_up(xmax, width)

        rows = int((top - bottom) / height)+1
        cols = int((right - left) / width)+1

        XleftOrigin = left
        XrightOrigin = left + width
        YtopOrigin = top
        YbottomOrigin = top - height
        polygons = []
        for i in range(cols):
            Ytop = YtopOrigin
            Ybottom = YbottomOrigin
            for j in range(rows):
                polygons.append(Polygon([(XleftOrigin, Ytop), (
                    XrightOrigin, Ytop), (XrightOrigin, Ybottom), (
                        XleftOrigin, Ybottom)]))
                Ytop = Ytop - height
                Ybottom = Ybottom - height
            XleftOrigin = XleftOrigin + width
            XrightOrigin = XrightOrigin + width

        grid = gpd.GeoDataFrame({'geometry': polygons})
        grid.crs = df.crs

        # Assign gridid
        numGrid = len(grid)
        grid['gridId'] = list(range(numGrid))

        # Identify gridId for each point
        points_identified = gpd.sjoin(df, grid, op='within')

        # group points by gridid and calculate mean Easting,
        # store it as dataframe
        # delete if field already exists
        if field in grid.columns:
            del grid[field]
        grouped = points_identified.groupby('gridId')[field].agg(summary)
        grouped_df = pd.DataFrame(grouped)

        new_grid = grid.join(grouped_df, on='gridId').fillna(0)
        new_grid['x_centroid'], new_grid['y_centroid'] = \
            new_grid.geometry.centroid.x, new_grid.geometry.centroid.y
        grid = new_grid
        return grid

    def plotAggregate(self, grid, field):
        """
        Plots the aggregated data on grid. Please call aggregateByGrid
            function before this step.

        Parameters
        ----------
        grid :polygon geodataframe
            The grid geodataframe with grid and aggregated data in a column.
                Grid shoud have grid id or equivalent unique ids
        field : string
            Fieldname with aggregated data

        Returns
        -------
        m : folium map object
            Folium map with openstreetmap as base.

        """
        # Prepare for grid plotting using folium
        grid.columns = [cols.replace('.', '_') for cols in grid.columns]
        field = field.replace('.', '_')
        # Convert grid id to string
        grid['gridId'] = grid['gridId'].astype(str)
        # only select grid with non zero values
        grid = grid[grid[field] > 0]

        # Convert data to geojson and csv
        atts = pd.DataFrame(grid.drop(columns=['geometry', 'x_centroid',
                                               'y_centroid']))
        grid.to_file("grids.geojson", driver='GeoJSON')
        atts.to_csv("attributes.csv", index=False)

        # load spatial and non-spatial data
        data_geojson_source = "grids.geojson"
        data_geojson = json.load(open(data_geojson_source))

        # Get coordiantes for map centre
        lat = grid.geometry.centroid.y.mean()
        lon = grid.geometry.centroid.x.mean()
        # Intialize a new folium map object
        m = folium.Map(location=[lat, lon], zoom_start=8,
                       tiles='OpenStreetMap')
        # Configure geojson layer
        folium.GeoJson(data_geojson).add_to(m)

        # add attribute data
        attribute_pd = pd.read_csv("attributes.csv")
        attribute = pd.DataFrame(attribute_pd)
        # Convert gridId to string to ensure it matches with gridId
        attribute['gridId'] = attribute['gridId'].astype(str)

        # construct color map
        minvalue = attribute[field].min()
        maxvalue = attribute[field].max()
        colormap_rn = linear.YlOrRd_09.scale(minvalue, maxvalue)

        # Create Dictionary for colormap
        population_dict_rn = attribute.set_index('gridId')[field]

        # create map
        folium.GeoJson(
           data_geojson,
           name='Choropleth map',
           style_function=lambda feature: {
                'fillColor': colormap_rn(population_dict_rn[feature[
                    'properties']['gridId']]),
                'color': 'black',
                'weight': 0.5,
                'dashArray': '5, 5',
                'fillOpacity': 0.5
                   },
           highlight_function=lambda feature: {'weight': 3, 'color': 'black',
                                               'fillOpacity': 1},
           tooltip=folium.features.GeoJsonTooltip(fields=[field],
                                                  aliases=[field])
        ).add_to(m)

        # format legend
        field = field.replace("_", " ")
        # add a legend
        colormap_rn.caption = '{value} per grid'.format(value=field)
        colormap_rn.add_to(m)

        # add a layer control
        folium.LayerControl().add_to(m)
        return m

    def flow_between_regions(self, data_mpd_df, from_region, to_region,
                             twoway):
        """ How many entities moved between from_region to to_region
            (one way or both ways)

        Keyword Arguments:
            data_mpd_df {GeoDataFrame} -- A Moving Pandas GeoDataFrame
                containing the track points
            from_region {Polygon} -- A shapely polygon as our Feautre
                of Interest (FOI) - 1
            to_region {Polygon} -- A shapely polygon as our Feautre
                of Interest (FOI) - 2
            twoways {Boolean} -- if two way or one regions are to be computed

        Returns:
            regional_trajectories -- A list of trajectories moving between
                provided regions
        """
        # Converting mpd gdf into a trajectory collection object
        traj_collection = mpd.TrajectoryCollection(data_mpd_df, 'track.id')

        regional_trajectories = []

        # To extract trajectories running between regions
        for traj in traj_collection.trajectories:
            if traj.get_start_location().intersects(from_region):
                if traj.get_end_location().intersects(to_region):
                    regional_trajectories.append(traj)
            if twoway:  # if two way is to be considered
                if traj.get_start_location().intersects(to_region):
                    if traj.get_end_location().intersects(from_region):
                        regional_trajectories.append(traj)

        if twoway:
            print("Found {} trajectories moving between provided regions with \
                following details:".format(len(regional_trajectories)))
        else:
            print("Found {} trajectories moving from 'from_region' to \
                'to_region' with following details:".format(
                    len(regional_trajectories)))

        lengths = []
        durations = []

        # To extract Stats related to Distance and Duration
        for row in regional_trajectories:
            lengths.append(round((row.get_length()/1000), 2))
            durations.append(row.get_duration().total_seconds())

        print("Average Distance: {} kms".format(round(mean(lengths), 2)))
        print("Maximum Distance: {} kms".format(max(lengths)))
        print("Average Duration: {} ".format(str(datetime.timedelta(
            seconds=round(mean(durations), 0)))))
        print("Maximum Duration: {} ".format(str(datetime.timedelta(
            seconds=round(max(durations), 0)))))

        # List of Trajectories between regions
        return regional_trajectories

    def temporal_filter_weekday(self, mpd_df, filterday):
        """ Applies temporal filter to the dataframe based on provided WEEKDAY

        Keyword Arguments:
            mpd_df {GeoDataFrame} -- A Moving Pandas GeoDataFrame containing
                the track points
            filterday {String} -- Provided day of the week

        Returns:
            result -- A Trajectory Collection Object with only trajectories
                from provided weekday
        """
        # Conversion of mpd geodataframe into Trajectory Collection Object
        # of Moving Pandas
        raw_collection = mpd.TrajectoryCollection(mpd_df, 'track.id',
                                                  min_length=1)

        # In case, a single trajectory span over two days, split trajectory
        # into two
        traj_collection = raw_collection.split_by_date('day')

        days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                4: "Friday", 5: "Saturday", 6: "Sunday"}

        # Loop over all trajectories in Trajectory Collection Object
        for traj in traj_collection.trajectories:
            # Extract the total number of column in each trajectory's dataframe
            numcolumns = len(traj.df.columns)

            # Extracting track begin time in datetime object
            temp_time = pd.to_datetime(traj.df['track.begin'],
                                       format='%Y-%m-%dT%H:%M:%SZ')

            # Insertion of two new rows for Formatted Time and Day of the Week
            traj.df.insert(numcolumns, 'Trajectory Time', temp_time)
            traj.df.insert(numcolumns+1, 'Day of Week', 'a')

            # Extracting the time of first row of trajectory df and assign
            # Day of the week to the whole column
            time_value = traj.df['Trajectory Time'][0]
            traj.df['Day of Week'] = days[time_value.weekday()]

        filterday_tracks = []
        # Loop over first row of all trajectories df and select track.id
        # satisfying DAY of the Week condition
        for traj in traj_collection.trajectories:
            if(traj.df['Day of Week'][0] == filterday):
                filterday_tracks.append(traj.df['track.id'][0])

        filtered = []
        # Loop over list of filtered track.ids and trajectories collection.
        # Filter trajectories with identified track.ids
        for f_track in filterday_tracks:
            for traj in traj_collection.trajectories:
                if(traj.df['track.id'][0] == f_track):
                    filtered.append(traj)
                    break

        # Creating a Trajectory Collection and assign filtered trajectories
        # to it as result
        result = copy(traj_collection)
        result.trajectories = filtered

        return result

    def temporal_filter_hours(self, mpd_df, from_time, to_time):
        """ Applies temporal filter to the dataframe based on provided HOURS duration

        Keyword Arguments:
            mpd_df {GeoDataFrame} -- A Moving Pandas GeoDataFrame containing
                the track points
            from_time {Integer} -- Starting Hour
            end_time {Integer} -- Ending Hour

        Returns:
            result -- A Trajectory Collection Object with only trajectories
                from provided hours duration
        """

        filtered = []

        # Conversion of mpd geodataframe into Trajectory Collection Object of
        # Moving Pandas
        raw_collection = mpd.TrajectoryCollection(mpd_df, 'track.id',
                                                  min_length=1)

        # In case, a single trajectory span over two days,
        # split trajectory into two
        traj_collection = raw_collection.split_by_date('day')

        for traj in traj_collection.trajectories:
            # Extracting data for each trajectory
            mydate = traj.df['track.begin'][0][0:10]
            # Converting given hour number to datetime string
            from_time_string = mydate + ' ' + str(from_time) + ':00:00'
            to_time_string = mydate + ' ' + str(to_time) + ':00:00'

            # Filter part of trajectory based on provided hours duration
            filt_segment = traj.df[from_time_string:to_time_string]

            if(len(filt_segment) > 0):
                filtered.append(mpd.Trajectory(filt_segment,
                                               traj.df['track.id']))

        # Creating a Trajectory Collection and assign filtered trajectories
        # to it as result
        result = copy(traj_collection)
        result.trajectories = filtered

        return result

    def temporal_filter_date(self, mpd_df, filterdate):
        """ Applies temporal filter to the dataframe based on provided DATE

        Keyword Arguments:
            mpd_df {GeoDataFrame} -- A Moving Pandas GeoDataFrame containing
                the track points
            filterdate {String} -- Date for Filter

        Returns:
            result -- A Trajectory Collection Object with only trajectories
                from provided DATE
        """

        # Conversion of mpd geodataframe into Trajectory Collection Object
        # of Moving Pandas
        raw_collection = mpd.TrajectoryCollection(mpd_df, 'track.id',
                                                  min_length=1)

        # In case, a single trajectory span over two days, split trajectory
        # into two
        traj_collection = raw_collection.split_by_date('day')

        filterday_tracks = []
        # Loop over first row of all trajectories df and select track.id
        # satisfying DATE condition
        for traj in traj_collection.trajectories:
            if(traj.df['track.begin'][0][0:10] == filterdate):
                filterday_tracks.append(traj.df['track.id'][0])

        filtered = []
        # Loop over list of filtered track.ids and trajectories collection.
        # Filter trajectories with identified track.ids
        for f_track in filterday_tracks:
            for traj in traj_collection.trajectories:
                if(traj.df['track.id'][0] == f_track):
                    filtered.append(traj)
                    break

        # Creating a Trajectory Collection and assign filtered trajectories to
        # it as result
        result = copy(traj_collection)
        result.trajectories = filtered

        return result
