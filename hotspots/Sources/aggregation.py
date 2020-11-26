import os
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
#import movingpandas as mpd
from copy import copy
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import json
from branca.colormap import linear
import enum
from geofeather import to_geofeather, from_geofeather


'''function from group -4 aggregated by grids'''
def aggregateByGrid(df,field,summary,gridSize):
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

'''plot aggregation in folium map'''
def plotAggregate(grid,field):
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
        attribute_pd=pd.read_csv("attributes.csv")
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
                   },
            highlight_function=lambda feature:{'weight':3,'color':'black','fillOpacity':1},
            tooltip=folium.features.GeoJsonTooltip(fields=[field],aliases=[field])
        ).add_to(m)
    
       #format legend
        field=field.replace("_"," ")
        # add a legend
        colormap_rn.caption = '{value} per grid'.format(value=field)
        colormap_rn.add_to(m)
    
        # add a layer control
        folium.LayerControl().add_to(m)