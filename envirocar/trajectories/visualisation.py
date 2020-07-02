
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt
import datetime
import folium
import random
import seaborn as sns
import pandas as pd
import plotly.express as px
import geopandas as gpd
# import movingpandas as mpd
# from statistics import mean
from shapely.geometry import Polygon, MultiPoint
import json
from branca.colormap import linear
# from copy import copy
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle


class Visualiser():
    def __init__(self):
        print("Initializing visualisation class")   # do we need anything?

    def st_cube_simple(self, points):
        """ To plot a space-time cube of one trajectory. Checks for the start time
            and calculates seconds passed from it for every next point

        Keyword Arguments:
            points {dataframe} -- A Pandas dataframe of a trajectory
        Returns:
            No Return
        """

        def seconds_from_start(x, start):
            date_time_obj = datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')
            seconds = (date_time_obj-start).total_seconds()
            return int(seconds)

        points['lat'] = points['geometry'].apply(lambda coord: coord.y)
        points['lng'] = points['geometry'].apply(lambda coord: coord.x)
        start_time = datetime.datetime.strptime(
            points.time.iloc[0], '%Y-%m-%dT%H:%M:%S')

        points['time_seconds'] = np.vectorize(seconds_from_start)(
            np.array(points.time.values.tolist()), start_time)

        # plot the space-time cube
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(points['lng'], points['lat'], points['time_seconds'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Seconds since start')
        fig.canvas.set_window_title('Space-Time Cube')
        plt.show()

    def plot_full_correlation(self, points_df):
        """ To plot a correlation matrix for all columns that contain word
            '.value' in their name

        Keyword Arguments:
            points_df {dataframe} -- A Pandas dataframe of a trajectory
        Returns:
            No Return
        """

        value_names = [s for s in points_df.columns if
                       '.value' in s]

        value_columns = [np.array(
            points_df[column].values.tolist()) for column
            in value_names]

        values_transposed = np.transpose(value_columns)

        values_df = pd.DataFrame(values_transposed)
        values_df.columns = value_names

        f, ax = plt.subplots(figsize=(10, 8))
        corr = values_df.corr()
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax)

    def plot_pair_correlation(self, points_df, column_1, column_2,
                              sort_by='id', regression=False):
        """ To plot a pairwise relationship in a dataset.
        Special case for the Acceleration values to see difference
        (if any) between accelerating and braking.

        Keyword Arguments:
            points_df {dataframe} -- A Pandas dataframe of a trajectory
            column_1, column_2 {string} -- names of 2 columns to analyse
            sort_by {string} -- 'id' or 'temperature'
            regression {boolean} -- defines which kind of plot to plot
        Returns:
            No Return
        """

        if (sort_by == 'temperature'):
            bins = [-10, 0, 5, 10, 20, 30, 40]
            copied = points_df.copy()
            copied['Intake Temperature.value'] = \
                copied['Intake Temperature.value'].astype(int)
            copied['binned_temp'] = pd.cut(copied['Intake Temperature.value'],
                                           bins)

            if (column_2 == "Acceleration.value" or
                    column_1 == "Acceleration.value"):
                df1 = copied[copied["Acceleration.value"] > 0]
                df2 = copied[copied["Acceleration.value"] < 0]

                if (regression):
                    sns.lmplot(x=column_1, y=column_2, hue='binned_temp',
                               data=df1, palette="viridis")
                    sns.lmplot(x=column_1, y=column_2, hue='binned_temp',
                               data=df2, palette="viridis")
                else:
                    sns.pairplot(df1, vars=[column_1, column_2],
                                 hue="binned_temp")
                    sns.pairplot(df2, vars=[column_1, column_2],
                                 hue="binned_temp")

            else:
                if (regression):
                    sns.lmplot(x=column_1, y=column_2, hue='binned_temp',
                               data=copied)
                else:
                    sns.pairplot(copied, vars=[column_1, column_2],
                                 hue="binned_temp")

        else:
            if (column_2 == "Acceleration.value" or
                    column_1 == "Acceleration.value"):
                df1 = points_df[points_df["Acceleration.value"] > 0]
                df2 = points_df[points_df["Acceleration.value"] < 0]

                if (regression):
                    sns.lmplot(x=column_1, y=column_2, hue='track.id',
                               data=df1, palette="viridis")
                    sns.lmplot(x=column_1, y=column_2, hue='track.id',
                               data=df2, palette="viridis")
                else:
                    sns.pairplot(df1, vars=[column_1, column_2],
                                 hue="track.id")
                    sns.pairplot(df2, vars=[column_1, column_2],
                                 hue="track.id")

            else:
                if (regression):
                    sns.lmplot(x=column_1, y=column_2, hue='track.id',
                               data=points_df, palette="viridis")
                else:
                    sns.pairplot(points_df, vars=[column_1, column_2],
                                 hue="track.id")

    def plot_distribution(self, points, column):
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [5, 5, 5]})

        sns.boxplot(x=points[column], ax=ax1)
        ax1.set_title('Boxplot')
        sns.kdeplot(points[column], shade=True, color="r", ax=ax2)
        ax2.set_title('Gaussian kernel density estimate')
        sns.distplot(points[column], kde=False, ax=ax3)
        ax3.set_title('Histogram')

        fig.tight_layout()
        plt.show()

    def create_map(self, trajectories):
        """ To create a Folium Map object (in case its not already available)

        Keyword Arguments:
            trajectories {mpd trajectory collection} -- A Moving Pandas
            Trajectory Collection

        Returns:
            map {folium map} -- Newly created map object
        """
        map_zoom_point = []
        map_zoom_point.append(trajectories[0].df['geometry'][0].y)
        map_zoom_point.append(trajectories[0].df['geometry'][0].x)
        map = folium.Map(location=[map_zoom_point[0], map_zoom_point[1]],
                         zoom_start=12, tiles='cartodbpositron')
        return map

    def plot_flows(self, flows, flow_map):
        """ To plot provided aggregated flows over the provided map

        Keyword Arguments:
            flows {mpd aggregated flows} -- A Moving Pandas Aggreagtion
                function output
            flow_map {folium map} -- Map over which trajectories are to be
                plotted
        Returns:
            No Return
        """
        index = 0
        # to extract coordiantes from "FLOWS"
        for row in range(0, len(flows)):
            my_poylyline = []
            mylng = flows.loc[index, 'geometry'].coords[0][0]
            mylat = flows.loc[index, 'geometry'].coords[0][1]
            my_poylyline.append([mylat, mylng])

            mylng = flows.loc[index, 'geometry'].coords[1][0]
            mylat = flows.loc[index, 'geometry'].coords[1][1]
            my_poylyline.append([mylat, mylng])

            # to plot point's coordinates over the map as polyline based on
            # weight
            myweight = int(flows.loc[index, 'weight'])
            my_line = folium.PolyLine(locations=my_poylyline,
                                      weight=round((myweight/2)))
            # as minimize very big weight number
            flow_map.add_child(my_line)

            index += 1

    def plot_point_values(self, points, value):
        """ To show points on a map

        Keyword Arguments:
            points {GeoDataFrame} -- points input
            value {string} -- column value to use for colouriing

        Returns:
            No Return
        """

        points['lat'] = points['geometry'].apply(lambda coord: coord.y)
        points['lng'] = points['geometry'].apply(lambda coord: coord.x)

        # Visualizing points by the desired value
        fig = px.scatter_mapbox(points, lat="lat", lon="lng", color=value,
                                title=value + " visualisation", zoom=8)
        fig.update_layout(mapbox_style="open-street-map",
                          margin={"r": 5, "t": 50, "l": 10, "b": 5})
        fig.show()

    def plot_region(self, region, region_map, region_color, label):
        """ To plot provided regions over the provided map

        Keyword Arguments:
            region {shapely Polygon} -- A shapely based Polygon
            region_map {folium map} -- Map over which trajectories are to be
                plotted
            region_color {string} -- Name of the Color in String
            label {String} -- Label for popup
        Returns:
            No Return
        """
        region_coords = []

        # to extract coordiantes from provided region
        index = 0
        for value in range(0, len(region.exterior.coords)):
            temp = []
            temp.append(region.exterior.coords[index][1])
            temp.append(region.exterior.coords[index][0])
            region_coords.append(temp)
            index += 1

        # to plot point's coordinates over the map as polygon
        region_plot = folium.Polygon(locations=region_coords,
                                     color=region_color, popup=label)
        region_map.add_child(region_plot)

    def plot_weeks_trajectory(self, weekwise_trajectory_collection,
                              trajectory_map, marker_radius):
        """ To iterate over list with weekwise trajectory collection and plot
            each over provided folium map object

        Keyword Arguments:
            weekwise_trajectory_collection {list of mpd trajectory collection}
                -- 7 indices respective of each day of the week
            trajectory_map {folium map} -- Map over which trajectories are to
                be plotted
            marker_radius {integer} -- Radius of each point marker (circle)
        Returns:
            No Return
        """

        # Dictionary to assign color based on a week day
        colors = {0: "crimson", 1: "blue", 2: "purple", 3: "yellow",
                  4: "orange", 5: "black", 6: "green"}

        day = 0
        for traj_day in weekwise_trajectory_collection:

            track_id = -1  # to store track id of each track for Pop Up

            trajectory_points = []  # to store coordiante points for each track
            traj_row = 0

            # if trajectory collection has atleast a single trajectory
            if(len(traj_day.trajectories) > 0):
                for traj in traj_day.trajectories:
                    point_row = 0
                    track_id = traj.df['track.id'][0]
                    for point in range(len(traj_day.trajectories[
                            traj_row].df)):
                        temp = []
                        temp.append(traj.df['geometry'][point_row].y)
                        temp.append(traj.df['geometry'][point_row].x)
                        trajectory_points.append(temp)
                        point_row += 1
                    traj_row += 1

                # Plotting day wise point's coordinate plot with a single
                # color and track id as popup
                for row in trajectory_points:
                    folium.Circle(radius=marker_radius, location=row,
                                  color=colors[day], popup=track_id).add_to(
                                      trajectory_map)

            day += 1

    def get_trajectories_coords(self, trajectories):
        """ To iterate over trajectory collection and return individual track points

        Keyword Arguments:
            trajectories {mpd trajectory collection} -- A Moving Pandas
                Trajectory Collection

        Returns:
            trajectory_list -- A list of two elements at each index,
                track_id & array of associated point's coordinates
        """

        trajectory_list = []

        for traj in trajectories:
            track_points = []

            # Extracting Point's coordinate for each trajectory
            for i in range(len(traj.df)):
                temp = []
                temp.append(traj.df['geometry'][i].y)
                temp.append(traj.df['geometry'][i].x)
                track_points.append(temp)

            # Extracting Track_Id for each trajectory
            track_id = []
            track_id.append(traj.df['track.id'][0])

            # Creating a list with [id,coordinates] for each individual
            # trajectory
            traj_temp = [track_id, track_points]
            trajectory_list.append(traj_temp)

        return trajectory_list

    def plot_trajectories(self, trajectory_collection, trajectory_map,
                          marker_radius):
        """ To iterate over trajectory collection and plot each over
        provided folium map object

        Keyword Arguments:
            trajectory_collection {mpd trajectory collection}
                -- A Moving Pandas Trajectory Collection
            trajectory_map {folium map} -- Map over which trajectories are
                to be plotted
            marker_radius {integer} -- Radius of each point marker (circle)
        Returns:
            No Return
        """

        # Function to get random hexcode to assign unique color to each
        # trajectory
        def get_hexcode_color():
            random_number = random.randint(0, 16777215)
            hex_number = str(hex(random_number))
            hex_number = '#' + hex_number[2:]
            return hex_number

        # Call to function to iterate over trajectory collection
        # and return individual track points
        traj_list = self.get_trajectories_coords(trajectory_collection)

        traj_index = 0
        for traj in traj_list:
            # Extracting Track_Id and Point's coordinate for each trajectory
            track_id = traj[0]
            track_points = traj[1]

            # Call to function to random color for this trajectory
            track_color = get_hexcode_color()

            # Plotting points of each trajectory with a single color
            point_index = 0
            for row in track_points:
                # Pop-Up will contain Track Id
                folium.Circle(radius=marker_radius, location=row,
                              color=track_color, popup=track_id).add_to(
                                  trajectory_map)
                point_index += 1

            traj_index += 1

    ##################################
    # RELATED TO WEEK WISE BAR GRAPH #

    def extract_daywise_lengths(self, weekly_trajectories):
        """ To iterate over list with weekwise trajectory collection and
            extract point's coordinates for day wise trajectories

        Keyword Arguments:
            weekly_trajectories {list of mpd trajectory collection}
                -- 7 indices respective of each day of the week
        Returns:
            day_length {list} -- list with total length for each day
        """
        days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                4: "Friday", 5: "Saturday", 6: "Sunday"}

        day_length = []  # to store total length for each day at each index
        day = 0

        for traj_day in range(len(weekly_trajectories)):
            temp = []

            # if trajectory collection has atleast a single trajectory
            if(len(weekly_trajectories[day].trajectories) > 0):
                traj_row = 0
                length_sum = 0  # to store total sum of track length for each
                # day's collection

                for traj in range(len(weekly_trajectories[day].trajectories)):
                    length_sum += round(weekly_trajectories[day].trajectories[
                        traj_row].df['track.length'][0], 2)
                    traj_row += 1

                temp.append(days[day])  # storing weekday name like Monday,
                # Tuesday etc at first index of list
                temp.append(length_sum)  # storing assocaited total length
                # at second index of list
                day_length.append(temp)

            else:
                temp.append(days[day])
                temp.append(0)
                day_length.append(temp)

            day += 1

        return day_length

    def extract_barplot_info(self, day_length):
        """ To extract information for matplotlib plot

        Keyword Arguments:
            day_length {list} -- list with total length for each day
        Returns:
            day, height, highest, highest_index, average {strings/integers}
                -- attributes required for plots
        """
        day = []
        height = []
        highest = 0
        highest_index = -1
        total = 0

        index = 0
        for row in day_length:
            day.append(row[0][:3])  # extracting name of day of the week
            # in form of Mon, Tue etc.
            track_length = round(row[1], 2)  # extracting total length
            # associated with each day rounded to 2 decimals
            height.append(track_length)

            # extracting the highest value out of 'total lengths' from all
            # weekdays
            if(track_length > highest):
                highest = track_length
                highest_index = index

            total += track_length
            index += 1

        average_value = total/7  # extracting average value out of
        # 'total lengths' from all weekdays

        average = []
        for row in day:
            average.append(average_value)  # a list of same value at each
        # index, just to plot a horizontal line in plot

        return day, height, highest, highest_index, average

    def plot_daywise_track(self, week_trajectories):
        """ To plot bar graphy of week day vs total length of that day
            (all tracks combined)

        Keyword Arguments:
            weekly_trajectories {list of mpd trajectory collection}
                -- 7 indices respective of each day of the week
        Returns:
            No Return
        """
        # Call to function to extract daywise lengths
        daywise_length = self.extract_daywise_lengths(week_trajectories)

        # Call to function to extract attributes for plot
        day, height, highest, highest_index, average = \
            self.extract_barplot_info(daywise_length)

        bar_plot = plt.bar(day, height, color=(0.1, 0.1, 0.1, 0.1),
                           edgecolor='blue')
        bar_plot[highest_index].set_edgecolor('r')
        plt.ylabel('Total Distance Travelled (Km)')

        axes2 = plt.twinx()
        axes2.set_ylim(0, highest+1)
        axes2.plot(day, average, color='b', label='Average Distance')

        plt.suptitle('Which day has a different movement pattern than others?')
        plt.legend()
        plt.show()

    def aggregateByGrid(df, field, summary, gridSize):
        """
        Aggregates the specified field with chosen summary type and user
            defined grid size. returns aggregated grids with summary

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

        # Get crs from data
        sourceCRS = df.crs
        targetCRS = {"init": "EPSG:3857"}
        # Reproject to Mercator\
        df = df.to_crs(targetCRS)
        # Get bounds
        xmin, ymin, xmax, ymax = df.total_bounds
        print(xmin, ymin, xmax, ymax)
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
                polygons.append(Polygon([(XleftOrigin, Ytop),
                                         (XrightOrigin, Ytop),
                                         (XrightOrigin, Ybottom),
                                         (XleftOrigin, Ybottom)]))
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
        grid = new_grid.to_crs(sourceCRS)
        summarized_field = summary+"_"+field
        final_grid = grid.rename(columns={field: summarized_field})
        final_grid = final_grid[final_grid[summarized_field] > 0].sort_values(
            by=summarized_field, ascending=False)
        final_grid[summarized_field] = round(final_grid[summarized_field], 1)
        final_grid['x_centroid'], final_grid['y_centroid'] = \
            final_grid.geometry.centroid.x, final_grid.geometry.centroid.y
        return final_grid

    def plotAggregate(grid, field):
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

        # Convert data to geojson and csv
        atts = pd.DataFrame(grid)
        grid.to_file("grids.geojson", driver='GeoJSON')
        atts.to_csv("attributes.csv", index=False)

        # load spatial and non-spatial data
        data_geojson_source = "grids.geojson"
        # data_geojson=gpd.read_file(data_geojson_source)
        data_geojson = json.load(open(data_geojson_source))

        # Get coordiantes for map centre
        lat = grid.geometry.centroid.y.mean()
        lon = grid.geometry.centroid.x.mean()
        # Intialize a new folium map object
        m = folium.Map(location=[lat, lon], zoom_start=10,
                       tiles='OpenStreetMap')

        # Configure geojson layer
        folium.GeoJson(data_geojson,
                       lambda feature: {'lineOpacity': 0.4,
                                        'color': 'black',
                                        'fillColor': None,
                                        'weight': 0.5,
                                        'fillOpacity': 0}).add_to(m)

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
                'lineOpacity': 0,
                'color': 'green',
                'fillColor': colormap_rn(
                    population_dict_rn[feature['properties']['gridId']]),
                'weight': 0,
                'fillOpacity': 0.6
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

    def spatioTemporalAggregation(df, field, summary, gridSize):
        """
        Aggregates the given field on hour and weekday basis.
        Prepares data for mosaic plot

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
        geodataframes: one each for larger grid and other for subgrids
            (for visualization purpose only)
            Aggregated grids with summary on it

        """
        def round_down(num, divisor):
            return floor(num / divisor) * divisor

        def round_up(num, divisor):
            return ceil(num / divisor) * divisor

        # Get crs from data
        sourceCRS = df.crs
        targetCRS = {'init': "epsg:3857"}
        # Reproject to Mercator\
        df = df.to_crs(targetCRS)

        # Get bounds
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
                polygons.append(Polygon(
                    [(XleftOrigin, Ytop), (XrightOrigin, Ytop),
                     (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)]))
                Ytop = Ytop - height
                Ybottom = Ybottom - height
            XleftOrigin = XleftOrigin + width
            XrightOrigin = XrightOrigin + width

        grid = gpd.GeoDataFrame({'geometry': polygons})
        grid.crs = (targetCRS)

        # Assign gridid
        numGrid = len(grid)
        grid['gridId'] = list(range(numGrid))

        # Identify gridId for each point

        df['hour'] = df['time'].apply(
            lambda x: datetime.datetime.strptime(
                x, '%Y-%m-%dT%H:%M:%S')).dt.hour
        df['weekday'] = df['time'].apply(
            lambda x: datetime.datetime.strptime(
                x, '%Y-%m-%dT%H:%M:%S')).dt.dayofweek
        points_identified = gpd.sjoin(df, grid, op='within')

        # group points by gridid and calculate mean Easting,
        # store it as dataframe
        # delete if field already exists
        if field in grid.columns:
            del grid[field]

        # Aggregate by weekday, hour and grid
        grouped = points_identified.groupby(
            ['gridId', 'weekday', 'hour']).agg({field: [summary]})
        grouped = grouped.reset_index()
        grouped.columns = grouped.columns.map("_".join)
        modified_fieldname = field+"_"+summary

        # Create Subgrids
        subgrid, mainGrid, rowNum, columnNum, value = [], [], [], [], []
        unikGrid = grouped['gridId_'].unique()
        for currentGrid in unikGrid:
            dataframe = grid[grid['gridId'] == currentGrid]
            xmin, ymin, xmax, ymax = dataframe.total_bounds
            xminn, xmaxx, yminn, ymaxx = xmin + \
                (xmax-xmin)*0.05, xmax-(xmax-xmin)*0.05, ymin + \
                (ymax-ymin)*0.05, ymax-(ymax-ymin)*0.05
            rowOffset = (ymaxx-yminn)/24.0
            colOffset = (xmaxx - xminn)/7.0
            for i in range(7):
                for j in range(24):
                    topy, bottomy, leftx, rightx = ymaxx-j*rowOffset, ymaxx - \
                        (j+1)*rowOffset, xminn+i * \
                        colOffset, xminn+(i+1)*colOffset
                    subgrid.append(
                        Polygon([(leftx, topy), (rightx, topy),
                                 (rightx, bottomy), (leftx, bottomy)]))
                    mainGrid.append(currentGrid)
                    rowNum.append(j)
                    columnNum.append(i)
                    if len(grouped[(grouped['gridId_'] == currentGrid)
                           & (grouped['weekday_'] == i)
                           & (grouped['hour_'] == j)]) != 0:
                        this_value = grouped[
                            (grouped['gridId_'] == currentGrid)
                            & (grouped['weekday_'] == i)
                            & (grouped['hour_'] == j)].iloc[0][
                                modified_fieldname]
                        value.append(this_value)
                    else:
                        value.append(np.nan)
        subgrid_gpd = gpd.GeoDataFrame({'geometry': subgrid})
        subgrid_gpd.crs = targetCRS
        # Reproject to Mercator\
        subgrid_gpd = subgrid_gpd.to_crs(sourceCRS)
        subgrid_gpd['gridId'] = mainGrid
        subgrid_gpd['Weekday'] = columnNum
        subgrid_gpd['hour'] = rowNum
        subgrid_gpd['gridId'] = subgrid_gpd.apply(lambda x: str(
            x['gridId'])+"_"+str(x['Weekday'])+"_"+str(x['hour']), axis=1)
        subgrid_gpd[modified_fieldname] = value
        subgrid_gpd = subgrid_gpd.dropna()
        grid = grid.to_crs(sourceCRS)
        grid = grid[grid['gridId'].isin(unikGrid)]
        return grid, subgrid_gpd
        # final_subgrid=subgrid_gpd[subgrid_gpd['value'].notnull()]
        # return final_subgrid

    def MosaicPlot(mainGrid, grid, field):
        """
        Performs spatio temporal aggregation of data on weekday and hour,
            and prepares mosaicplot.

        Parameters
        ----------
        mainGrid :polygon geodataframe
            The grid geodataframe with grid and aggregated data in a column.
            Grid shoud have grid id or equivalent unique ids
        grid: Small subgrids, prepared for visualization purpose
        only represents an hour of a weekday
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

        # Convert maingrid,subgrid to geojson and csv
        mainGrid.to_file("mainGrids.geojson", driver='GeoJSON')
        atts = pd.DataFrame(grid)
        grid.to_file("grids.geojson", driver='GeoJSON')
        atts.to_csv("attributes.csv", index=False)

        # load spatial and non-spatial data
        data_geojson_source = "grids.geojson"
        # data_geojson=gpd.read_file(data_geojson_source)
        data_geojson = json.load(open(data_geojson_source))

        # load spatial and non-spatial data
        grid_geojson_source = "mainGrids.geojson"
        mainGrid_geojson = json.load(open(grid_geojson_source))

        # Get coordiantes for map centre
        lat = grid.geometry.centroid.y.mean()
        lon = grid.geometry.centroid.x.mean()
        # Intialize a new folium map object
        m = folium.Map(location=[lat, lon],
                       zoom_start=10, tiles='Stamen Toner')

        # Configure geojson layer
        # style = {'fillColor': '#f5f5f5', 'lineColor': '#ffffbf'}
        # polygon = folium.GeoJson(gjson, style_function = \
        # lambda x: style).add_to(m)
        # def style_function():
        # return {'fillColor': '#00FFFFFF', 'lineColor': '#00FFFFFF'}
        # folium.GeoJson(data_geojson).add_to(m)
        folium.GeoJson(mainGrid_geojson,
                       lambda feature: {'lineOpacity': 0.4,
                                        'color': '#00ddbb',
                                        'fillColor': None,
                                        'weight': 2,
                                        'fillOpacity': 0}).add_to(m)

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
                'lineOpacity': 0,
                'color': 'green',
                'fillColor': colormap_rn(population_dict_rn[
                    feature['properties']['gridId']]),
                'weight': 0,
                'fillOpacity': 0.9
            },
            highlight_function=lambda feature: {
                'weight': 3, 'color': 'black', 'fillOpacity': 1},
            tooltip=folium.features.GeoJsonTooltip(fields=['Weekday', 'hour',
                                                           field])).add_to(m)

        # format legend
        field = field.replace("_", " ")
        # add a legend
        colormap_rn.caption = '{value} per grid by weekday and hour'.format(
            value=field)
        colormap_rn.add_to(m)

        # add a layer control
        folium.LayerControl().add_to(m)
        return m
        # Aggregate data by weekday and hour

    def aggregateHourly(df, field, summary):
        """
        Aggregates the whole data by weekday and hour as preparation step for
            mosaic plot

        Parameters
        ----------
        df : GeoDataFrame
            The dataset of points to be summarized
        field : STRING
            The field in input dataframe to be summarized
        summary : String
            The type of aggregation to be used.eg. mean, median,

        Returns
        -------
        dayhourAggregate : dataframe
            Aggregated Data by weekday and time

        """
        # extract date and time from timestamp
        df['hour'] = df['time'].apply(
            lambda x: datetime.datetime.strptime(
                x, '%Y-%m-%dT%H:%M:%S')).dt.hour
        df['weekday'] = df['time'].apply(
            lambda x: datetime.datetime.strptime(
                x, '%Y-%m-%dT%H:%M:%S')).dt.dayofweek
        # Aggregate by weekday and hour
        dayhourAggregate = df.groupby(
            ['weekday', 'hour']).agg({field: [summary]})
        dayhourAggregate = dayhourAggregate.reset_index()
        dayhourAggregate.columns = dayhourAggregate.columns.map("_".join)
        return dayhourAggregate

    def OriginAndDestination(df):
        """
        Return dataframe for origin and destinations for tracks
            by their trackid

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        origin : TYPE
            DESCRIPTION.
        destination : TYPE
            DESCRIPTION.

        """
        track_list = list(df['track.id'].unique())
        origin, destination = gpd.GeoDataFrame(), gpd.GeoDataFrame()
        for track in track_list:
            selected_tracks = df[df['track.id'] == track]
            current_origin = selected_tracks[selected_tracks['time']
                                             == selected_tracks['time'].min()]
            current_destination = selected_tracks[selected_tracks['time']
                                                  == selected_tracks[
                                                      'time'].max()]
            origin = origin.append(current_origin)
            destination = destination.append(current_destination)
        return origin, destination

    def getClusters(positions, distanceKM, min_samples=5):
        """
        Returns the clusters from the points based on provided data to no. of
            clusters based on DBScan Algorithm

        Parameters
        ----------
        positions : Geodataframe object
           Geodataframe with positions to be clustered
        distanceKM : Float
            Epsilon parameters fo dbscan algorithm in km. or, distance for
                clustering of points
        min_samples : Integer, optional
            DESCRIPTION. Minimum no. of points required to form cluster.
                If 1 is set,each individual will form their own cluster
                The default is 5.

        Returns
        -------
        Dataframe
            The dataframe with cluster centres co-ordinates and no. of points
                on the cluster.

        """
        def get_centermost_point(cluster):
            centroid = (MultiPoint(cluster).centroid.x,
                        MultiPoint(cluster).centroid.y)
            centermost_point = min(
                cluster, key=lambda point: great_circle(point, centroid).m)
            return tuple(centermost_point)
        df = positions.to_crs({'init': 'epsg:4326'})
        lon = df.geometry.x
        lat = df.geometry.y
        origin_pt = pd.DataFrame()
        # Populate lat lon to dataframe
        origin_pt['lat'] = lat
        origin_pt['lon'] = lon
        # add index to data
        coords = origin_pt.to_numpy()
        origin_pt.index = [i for i in range(len(lat))]
        #
        # Convert Data to projected and perform clustering
        kms_per_radian = 6371.0088
        epsilon = distanceKM / kms_per_radian
        db = DBSCAN(eps=epsilon, min_samples=min_samples,
                    algorithm='ball_tree', metric='haversine').fit(
                        np.radians(coords))
        cluster_labels = db.labels_
        validClusters = []
        for cluster in cluster_labels:
            if cluster != -1:
                validClusters.append(cluster)
        num_clusters = len(set(validClusters))
        clusters = pd.Series([coords[cluster_labels == n]
                              for n in range(num_clusters)])
        # Assigining clusterId to each point
        origin_pt['clusterId'] = cluster_labels
        # Identify cluster Centres
        centermost_points = clusters.map(get_centermost_point)

        # Create Geodataframe with attributes for cluster centroids
        clusterId = [i for i in range(len(centermost_points))]
        centroidLat = [centermost_points[i][0]
                       for i in range(len(centermost_points))]
        centroidLon = [centermost_points[i][1]
                       for i in range(len(centermost_points))]
        clusterSize = [len(origin_pt[origin_pt['clusterId'] == i])
                       for i in range(len(centermost_points))]
        # Create dataframe for cluster centers
        clusterCentres_df = pd.DataFrame(
            {'clusterId': clusterId, 'clusterLat': centroidLat,
             'clusterLon': centroidLon, 'clusterSize': clusterSize})
        clusterCentres = gpd.GeoDataFrame(clusterCentres_df,
                                          geometry=gpd.points_from_xy(
                                              clusterCentres_df.clusterLon,
                                              clusterCentres_df.clusterLat))
        return clusterCentres

    def showClusters(clusterCentres, track):
        """
        Shows the cluster of the datasets along with original tracks

        Parameters
        ----------
        clusterCentres : Geodataframe
            The geodataframe object with details of clusterCenters.
            Obtained as processing by getClusters fucntion
        track : Geodataframe
            The points geodataframe to be shown on map alongwith clusters.
            For visualization only

        Returns
        -------
        m : folium map-type object
            The map with source data and clusters overlaid

        """
        # Make an empty map
        lat = clusterCentres.geometry.y.mean()
        lon = clusterCentres.geometry.x.mean()
        clusterList = list(clusterCentres['clusterSize'])
        m = folium.Map(location=[lat, lon],
                       tiles="openstreetmap", zoom_start=12)

        # add points from track
        for i in range(0, len(track)):
            lat = track.iloc[i].geometry.y
            lon = track.iloc[i].geometry.x
            folium.Circle(
                location=[lat, lon],
                radius=0.05,
                color='black',
                weight=2,
                fill=True, opacity=0.5,
                fill_color='black',
            ).add_to(m)

            # add marker one by one on the map
        for i in range(0, len(clusterCentres)):
            folium.Circle(
                location=[clusterCentres.iloc[i]['clusterLat'],
                          clusterCentres.iloc[i]['clusterLon']],
                popup=clusterList[i],
                radius=clusterList[i]*10,
                color='red',
                weight=2,
                fill=True,
                fill_color='red'
            ).add_to(m)
        return m
