import numpy as np
import matplotlib.pyplot as plt
import datetime


class Visualiser():
    def __init__(self):
        print("Initializing visualisation class")   # do we need anything?

    def st_cube_simple(self, points):
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
