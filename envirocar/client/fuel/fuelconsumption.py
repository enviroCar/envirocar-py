import numpy as np
import pandas as pd
import geopandas as gpd
import time
import osmnx as ox



from ..request_param import RequestParam, BboxSelector, TimeSelector, car, road
from ..download_client import DownloadClient
from ..required_functions import generate_parms, request, distance, gradient, interpolation, fuel_consumptions, engine_power


class consumption():

    def fuel_consumption(car:car, osmbox:BboxSelector, track_df = gpd.GeoDataFrame()):
        # Request elevation values from an open source (opentopodata.org)
        url = 'https://api.opentopodata.org/v1/eudem25m?locations='
        tracks = pd.DataFrame(columns=track_df.columns)

        # Add surface in get attribute method
        ox.settings.useful_tags_way = ['bridge', 'tunnel', 'oneway', 'lanes', 'ref', 'name','highway', 'maxspeed', 'service', 'access', 'area','landuse', 'width', 'est_width', 'junction', 'surface']
        # get the osm graph for the same area
        G = ox.graph_from_bbox(osmbox.max_y, osmbox.min_y, osmbox.max_x, osmbox.min_x, network_type='drive')  

        # get the number of routes in the dataframe
        n = 0
        a = track_df['track.id'].value_counts()
        for i in a:
            n+=1
        # loop through the dataframe and get the elevation from open source for each record in each routes   
        for i in range (0,n):
            one_track_id = track_df['track.id'].unique()[i]
            one_track = track_df[track_df['track.id'] == one_track_id]
            # estimate the len of data
            batch = [int(len(one_track)/100),len(one_track)%100]
            elevation=[]
            # get elevation
            for i in range(batch[0]+1):
                #create requeest 100 parameter
                s = i*100
                e = (i+1)*100
                if i<batch[0]+1:
                    --e
                else:
                    e= e+batch[1]
                #check the parameters (s) not excced the lenght of the track
                if s >= len(one_track):
                    break
                #creat the request
                parms= generate_parms(one_track,s,e)
                #send the request and get the results
                access= url+parms
                part = request(access)
                if part==None:
                    part=[np.nan]*(e+1-s)
                #put the results in a list
                elevation.extend(part)
                time.sleep(1)
            one_track['elevation']=elevation
            # Filterout the null value and use GPS altitude to fill them
            temp=one_track[one_track['elevation'].isnull()==True]
            if len(temp)> 0:
                for i in temp.index:
                    one_track.loc[i,'elevation']=one_track.loc[i,'GPS Altitude.value']
            # Match the graph with osm and get maxspeed & surface attriubutes
            ox.settings.useful_tags_way = ["maxspeed","surface"]
            for i in one_track.index:
                lat= one_track.loc[i,'geometry'].y
                lng= one_track.loc[i,'geometry'].x
                x = (ox.get_nearest_edge(G, (lat, lng)))
                p = [x[0],x[1]]
                a = ox.utils_graph.get_route_edge_attributes(G, p)
                dic = a[0]
                # check if the edge has maximum speed value if not then use the value of previous point
                if "maxspeed" in dic:
                    one_track.loc[i,"maxspeed"] = dic["maxspeed"]
                else:
                    if i > 0:
                        m = one_track.loc[i-1,"maxspeed"]
                        one_track.loc[i,"maxspeed"] = m
                        
                # check if the edge has a surface value, if not then use the value of previous point    
                if "surface" in dic:
                    one_track.loc[i,"surface"] = dic["surface"]
                else:
                    if i > 0:
                        s = one_track.loc[i-1,"surface"]
                        one_track.loc[i,"surface"] =  s
                    

                # get the rolling resistance cofficient
                if one_track.loc[i, 'surface'] == "asphalt":
                    one_track.loc[i, 'rolling_resistance'] = 0.02 # source: engineeringtoolbox.com
                elif one_track.loc[i, 'surface'] == "cobblestone":
                    one_track.loc[i, 'rolling_resistance'] = 0.015 # source: engineeringtoolbox.com
                elif one_track.loc[i, 'surface'] == "paving_stones":
                    one_track.loc[i, 'rolling_resistance'] = 0.033 # source: The Automotive Chassis book
                else:
                    one_track.loc[i, 'rolling_resistance'] = 0.02 
            #loop through the dataframe and calculate the gradient
            for i in one_track.index:
                if (i == len(one_track)-1):
                    break
                # Get the coordinates of each point
                lat1= one_track.loc[i,'geometry'].y
                lat2= one_track.loc[i+1,'geometry'].y
                lon1= one_track.loc[i,'geometry'].x
                lon2= one_track.loc[i+1,'geometry'].x

                #calculate the elevation difference between the two points
                heightdiff = one_track.loc[i+1,'elevation'] - one_track.loc[i,'elevation']

                #calculate the distance between the two points/ and the accumulated distance
                one_track.loc[i+1,'seg_distance']= distance(lon1,lon2,lat1,lat2)
                if i == 0:
                    one_track.loc[i,'total_distance'] = 0
                    one_track.loc[i+1,'total_distance'] = one_track.loc[i+1,'seg_distance']
                else:
                    one_track.loc[i+1,'total_distance']= one_track.loc[i+1,'seg_distance'] + one_track.loc[i,'total_distance']
                grade = gradient(heightdiff,one_track.loc[i+1,'seg_distance'])
                one_track.loc[i,'gradient']= grade

            ## Add interval time
            #get the timestamp column from the track
            track_time = one_track[['time']]
            #convert the timestamp column to sereis
            time_track = track_time.iloc[:,0]
            #convert the timestamp to data time
            x = pd.to_datetime(time_track)
            x_list = x.tolist()
            #calculate the time difference in second, and put it in a list
            time_interval = []
            for i in range(len(x_list)):
                if i == 0 :
                    time_in_second = 0
                    one_track.loc[i,'accumulate_time_interval'] = time_in_second
                else:
                    time_in_second = ((x_list[i] - x_list[i-1]).total_seconds())
                    one_track.loc[i,'accumulate_time_interval'] = time_in_second + one_track.loc[i-1,'accumulate_time_interval']
                    
                time_interval.append(time_in_second)
            # add the time interval list to the track
            one_track['time_interval']=time_interval
        
            # Convert the speed unit to m/s
            for i in one_track.index:
                one_track.loc[i, 'speed'] = one_track.loc[i, 'GPS Speed.value'] / 3.6

            # calculate the acceleration

            for i in one_track.index:
                if (i == len(one_track)-1):
                    break
                else:
                    one_track.loc[i, 'Acceleration'] = (one_track.loc[i+1, 'speed'] - one_track.loc[i, 'speed'])/one_track.loc[i,'time_interval']

            ## Calculates Engine Power for general car
            for i in one_track.index:
                ep = engine_power(car,one_track.rolling_resistance[i],one_track.gradient[i],one_track.speed[i],one_track.Acceleration[i])
                if ep[0] < 0:
                    one_track.loc[i, 'engine_power'] = car.P_idle
                    one_track.loc[i, 'driving_resistance'] = ep[1]
                else:
                    one_track.loc[i, 'engine_power'] = ep[0]
                    one_track.loc[i, 'driving_resistance'] = ep[1]
            ## Driving resistance
            for i in one_track.index:
                res = one_track.loc[i, 'driving_resistance']
                if (res >= 2000 or res <= -2000):
                    one_track.loc[i,'efficiency'] = 0.3
                else:
                    one_track.loc[i, 'efficiency'] = interpolation(res)

            ## Fuel consumption/CO2 emissions for General car (gasoline)
            for i in one_track.index:
                car_cons = fuel_consumptions(one_track.engine_power[i],car, one_track.loc[i,'efficiency'])
                one_track.loc[i, 'Consumption_Gasoline'] = car_cons   ## liters / hour
                one_track.loc[i, 'CO2_Gasoline'] = car_cons * car.H_g      ## kg Co2 / hour
            
            tracks = pd.concat([tracks, one_track])
            tracks = tracks[['elevation','maxspeed','Acceleration','surface','rolling_resistance','gradient','efficiency','Consumption_Gasoline','CO2_Gasoline']]

            
        return tracks

