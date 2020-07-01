import numpy as np
import matplotlib.pyplot as plt
import datetime
import folium
import random
import seaborn as sns
import pandas as pd
import plotly.express as px
# import movingpandas as mpd
# from statistics import mean
# from shapely.geometry import Point, LineString, Polygon


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
