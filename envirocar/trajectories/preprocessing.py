import os
from math import floor, ceil
from scipy import interpolate
from statistics import mean
import pandas as pd
import numpy as np
import datetime
import folium
import movingpandas as mpd
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import json
from branca.colormap import linear

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


    def aggregate(self, track_df, MIN_LENGTH, MIN_GAP, MAX_DISTANCE, MIN_DISTANCE, MIN_STOP_DURATION ):
        """ Transforms to Moving Pandas, Converts into Trajectories, Ignore small trajectories and return Aggregated Flows

        Keyword Arguments:
            track_df {GeoDataFrame} -- A Moving Pandas GeoDataFrame containing the track points
            MIN_LENGTH {integer} -- Minimum Length of a Trajectory (to be considered as a Trajectory)
            MIN_GAP {integer} -- Minimum Gap (in minutes) for splitting single Trajectory into more 
            MAX_DISTANCE {integer} -- Maximum distance between significant points 
            MIN_DISTANCE {integer} -- Minimum distance between significant points 
            MIN_STOP_DURATION {integer} -- Minimum duration (in minutes) required for stop detection

        Returns:
            flows -- A GeoDataFrame containing Aggregared Flows (linestrings)
        """

        # Using MPD function to convert trajectory points into actual trajectories  
        traj_collection = mpd.TrajectoryCollection(track_df, 'track.id', min_length=MIN_LENGTH)
        print("Finished creating {} trajectories".format(len(traj_collection)))

        # Using MPD function to Split Trajectories based on time gap between records to extract Trips
        trips = traj_collection.split_by_observation_gap(datetime.timedelta(minutes=MIN_GAP))
        print("Extracted {} individual trips from {} continuous vehicle tracks".format(len(trips), len(traj_collection)))

        # Using MPD function to Aggregate Trajectories
        aggregator = mpd.TrajectoryCollectionAggregator(trips, max_distance=MAX_DISTANCE, min_distance=MIN_DISTANCE, min_stop_duration=datetime.timedelta(minutes=MIN_STOP_DURATION))
        #pts = aggregator.get_significant_points_gdf()
        #clusters = aggregator.get_clusters_gdf()
        flows = aggregator.get_flows_gdf()
        return flows
           
    def aggregateByGrid(self,df,field,summary,gridSize):
        """
        Aggregates the specified field with chosen summary type and user defined grid size. returns aggregated grids with summary

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
        xmin,ymin,xmax,ymax =  df.total_bounds
        height,width=gridSize,gridSize
        top,left=round_up(ymax,height),round_down(xmin,width)
        bottom,right=round_down(ymin,height),round_up(xmax,width)
    
    
        rows = int((top -bottom) /  height)+1
        cols = int((right -left) / width)+1
    
        XleftOrigin = left
        XrightOrigin = left + width
        YtopOrigin = top
        YbottomOrigin = top- height
        polygons = []
        for i in range(cols):
            Ytop = YtopOrigin
            Ybottom =YbottomOrigin
            for j in range(rows):
                polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)]))
                Ytop = Ytop - height
                Ybottom = Ybottom - height
            XleftOrigin = XleftOrigin + width
            XrightOrigin = XrightOrigin + width
    
        grid = gpd.GeoDataFrame({'geometry':polygons})
        grid.crs=df.crs
       
        #Assign gridid
        numGrid=len(grid)
        grid['gridId']=list(range(numGrid))
    
        #Identify gridId for each point
        points_identified= gpd.sjoin(df,grid,op='within')
    
        #group points by gridid and calculate mean Easting, store it as dataframe
        #delete if field already exists
        if field in grid.columns:
            del grid[field]
        grouped = points_identified.groupby('gridId')[field].agg(summary)
        grouped_df=pd.DataFrame(grouped)
        
        new_grid=grid.join(grouped_df, on='gridId').fillna(0)
        new_grid['x_centroid'],new_grid['y_centroid']=new_grid.geometry.centroid.x,new_grid.geometry.centroid.y
        grid=new_grid
        return grid
    
    def plotAggregate(self,grid,field):
        """
        Plots the aggregated data on grid. Please call aggregateByGrid function before this step.

        Parameters
        ----------
        grid :polygon geodataframe
            The grid geodataframe with grid and aggregated data in a column. Grid shoud have grid id or equivalent unique ids
        field : string
            Fieldname with aggregated data

        Returns
        -------
        m : folium map object
            Folium map with openstreetmap as base.

        """
        #Prepare for grid plotting using folium
        grid.columns=[cols.replace('.', '_') for cols in grid.columns]
        field=field.replace('.','_')
        #Convert grid id to string
        grid['gridId']=grid['gridId'].astype(str)
        #only select grid with non zero values
        grid=grid[grid[field]>0]
        
        #Convert data to geojson and csv 
        atts=pd.DataFrame(grid.drop(columns=['geometry','x_centroid','y_centroid']))
        grid.to_file("grids.geojson", driver='GeoJSON')
        atts.to_csv("attributes.csv", index=False)
        
        #load spatial and non-spatial data
        data_geojson_source="grids.geojson"
        data_geojson=json.load(open(data_geojson_source))
        
        #Get coordiantes for map centre
        lat=grid.geometry.centroid.y.mean()
        lon=grid.geometry.centroid.x.mean()
        #Intialize a new folium map object
        m = folium.Map(location=[lat,lon],zoom_start=8,tiles='OpenStreetMap')
        # Configure geojson layer
        folium.GeoJson(data_geojson).add_to(m)
    
        #add attribute data
        attribute_pd=pd.read_csv("C:\\Users\\DELL\\Desktop\\attributes.csv")
        attribute=pd.DataFrame(attribute_pd)
        #Convert gridId to string to ensure it matches with gridId
        attribute['gridId']=attribute['gridId'].astype(str)
        
        # construct color map
        minvalue=attribute[field].min()
        maxvalue=attribute[field].max()
        colormap_rn = linear.YlOrRd_09.scale(minvalue,maxvalue)
    
        #Create Dictionary for colormap
        population_dict_rn = attribute.set_index('gridId')[field]
    
        #create map
        folium.GeoJson(
           data_geojson,
            name='Choropleth map',
            style_function=lambda feature: {
                'fillColor': colormap_rn(population_dict_rn[feature['properties']['gridId']]),
                'color': 'black',
                'weight': 0.5,
                'dashArray': '5, 5',
                'fillOpacity':0.5
                   }
        ).add_to(m)
    
       #format legend
        field=field.replace("_"," ")
        # add a legend
        colormap_rn.caption = '{value} per grid'.format(value=field)
        colormap_rn.add_to(m)
    
        # add a layer control
        folium.LayerControl().add_to(m)
        return m

    def flow_between_regions(self, data_mpd_df, from_region, to_region, twoway):
        """ How many entities moved between from_region to to_region (one way or both ways) 

        Keyword Arguments:
            data_mpd_df {GeoDataFrame} -- A Moving Pandas GeoDataFrame containing the track points
            from_region {Polygon} -- A shapely polygon as our Feautre of Interest (FOI) - 1 
            to_region {Polygon} -- A shapely polygon as our Feautre of Interest (FOI) - 2 
            twoways {Boolean} -- if two way or one regions are to be computed

        Returns:
            regional_trajectories -- A list of trajectories moving between provided regions
        """
        # Converting mpd gdf into a trajectory collection object
        traj_collection = mpd.TrajectoryCollection(data_mpd_df, 'track.id')

        regional_trajectories = []

        # To extract trajectories running between regions
        for traj in traj_collection.trajectories:
            if traj.get_start_location().intersects(from_region):
                if traj.get_end_location().intersects(to_region):
                    regional_trajectories.append(traj)
            if twoway: #if two way is to be considered
                if traj.get_start_location().intersects(to_region):
                    if traj.get_end_location().intersects(from_region):
                        regional_trajectories.append(traj)

        if twoway:
            print("Found {} trajectories moving between provided regions with following details:".format(len(regional_trajectories)))
        else:
            print("Found {} trajectories moving from 'from_region' to 'to_region' with following details:".format(len(regional_trajectories)))

        index = 0
        lengths = []
        durations = []

        # To extract Stats related to Distance and Duration
        for row in regional_trajectories:
            lengths.append(round((regional_trajectories[index].get_length()/1000), 2))
            durations.append(regional_trajectories[index].get_duration().total_seconds())
            index +=1

        print("Average Distance: {} kms".format(round(mean(lengths),2)))
        print("Maximum Distance: {} kms".format(max(lengths)))
        print("Average Duration: {} ".format(str(datetime.timedelta(seconds = round(mean(durations),0)))))
        print("Maximum Duration: {} ".format(str(datetime.timedelta(seconds = round(max(durations),0)))))

        # List of Trajectories between regions
        return regional_trajectories 

    def cluster(self, points_mp):
        # TODO clustering of points here
        return 'Clustering function was called. Substitute this string with clustering result'
        