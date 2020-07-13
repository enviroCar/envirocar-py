import requests as req,json
from scipy import interpolate
import numpy as np
import math 
## Reqquired functions to calcilar the fuel consumption/CO2 emmisions

## Retrive elevation values from an open soure - up to 100000 values per day
def generate_parms(one_track,s,e):
    lats= list(one_track[s:e]['geometry'].y)
    lngs = list(one_track[s:e]['geometry'].x)
    track_coords = [c for c in zip(lats, lngs)]
    format_str=list(map(lambda x : str(x[0])+','+str(x[1])+'|', track_coords))
    concat_str = ''.join(format_str)
    return concat_str

def request(link):
    elevation = req.request('GET',link)
    results = elevation.json()['results']
    h = list(map(lambda x : x['elevation'], results))
    return h

## Calculate he ditance between the two gps record/
def distance(lon1,lon2,lat1,lat2):
    b = 69.1 * (lat2 - lat1)
    e = 69.1 * (lon2 - lon1) * np.cos(lat1/57.3)
    d = math.sqrt((b ** 2) + (e ** 2)) * 1609.344
    return d

## Calculate the gradeint on the segment    
def gradient(height,distance):
    return height/distance

## Calculate the efficiency value
def interpolation(x):
    A = [-2000, 2000]
    B = [0.1, 0.4]
    f = interpolate.interp1d(A, B)
    efficiency = f(x)
    return efficiency

## Define engine power (KW)
def engine_power(car,Cr,gradient,speed,acceleration):
    g = 9.81 #Gravitational acceleration "m/s²"
    P_air = 1.2 # Air mass density "kg per m³"
    if speed > 0:
        power =speed*(0.5*car.Cw*car.A*P_air*pow(speed,2) #driving resistance
                      +car.m*g*Cr*np.cos(gradient) #rolling resistence
                      +car.m*g*np.sin(gradient) # climbing resistance
                      +car.m*+acceleration) # inertial resistance
        return [power/1000, power/speed]
    else:
        resistance =(0.5*car.Cw*car.A*P_air*pow(speed,2) #driving resistance
                     +car.m*g*Cr*np.cos(gradient) #rolling resistence
                     +car.m*g*np.sin(gradient) # climbing resistance
                     +car.m*+acceleration) # inertial resistance
        return [car.P_idle, resistance]
    
## estimate the fuel consumtion 
def fuel_consumptions(eng_pow, car, efc):
    consumption = eng_pow / (car.H_g * efc)
    return consumption


