# import numpy as np
import similaritymeasures
# import matplotlib.pyplot as plt


class TrackSimilarity():

    def __init__(self):
        print("Initializing TrackSimilarity class")

    def similarity(self, method, trajectoryA, trajectoryB):

        """ Compute similarity measures using the similaritymeasures
        https://pypi.org/project/similaritymeasures/

        Keyword Arguments:
            method {string}     -- Name of the method to compute similarity
                                pcm: Partial Curve Mapping
                                frechet_dist: Discrete Frechet distance
                                area_between_two_curves: Area method
                                curve_length_measure: Curve Length
                                dtw: Dynamic Time Warping

            trajectoryA {envirocar trajectory}   -- Envirocar trajectory
            trajectoryB {envirocar trajectory}   -- Envirocar trajectory

        Returns:
            similarity -- Float value (0,1) corresponding to the computed
                similarity. Values close to 1 correspond to high similarity
            dtw_matrix (optional) -- Only for the Dynamic Time Warping the
                method returns the calculation matrix.
        """

        print("Similarity between Track", trajectoryA.id, "& Track",
              trajectoryB.id, "using", str(method), "method:")

        methods = ['pcm', 'frechet_dist', 'area_between_two_curves',
                   'curve_length_measure', 'dtw']

        trajA_np = trajectoryA.get_coordinates()
        trajB_np = trajectoryB.get_coordinates()

        if(method not in methods):
            raise RuntimeError(
                    'Method not available')
        else:
            similarity_method = getattr(similaritymeasures, method)

            if(method == 'dtw'):
                similarity, dtw_matrix = 1/(1+similarity_method(
                    trajA_np, trajB_np))
                print(similarity)
                return similarity, dtw_matrix
            else:
                similarity = 1/(1+similarity_method(trajA_np, trajB_np))
                print(similarity)
                return similarity
