import numpy as np
import pandas as pd
import similaritymeasures
from math import factorial
import matplotlib.pyplot as plt
from itertools import combinations
from timeit import default_timer as timer


def track_similarity(trajA, trajB, method):

    """ Compute similarity measures using the similaritymeasures package
        https://pypi.org/project/similaritymeasures/

        Keyword Arguments:
        trajA {movingpandas trajectory}  -- movingpandas trajectory
        trajB {movingpandas trajectory}  -- movingpandas trajectory
        method {string}                  -- Name of the method to compute
                                                similarity
                                            pcm: Partial Curve Mapping
                                            frechet_dist: Discrete Frechet
                                                distance
                                            area_between_two_curves: Area
                                                method
                                            curve_length_measure: Curve Length
                                            dtw: Dynamic Time Warping


        Returns:
            similarity -- Float value (0,1) corresponding to the computed
                similarity. Values close to 1 correspond to high similarity
        """

    # trajA_id = trajA.df['track.id'].unique()[0]
    # trajB_id = trajB.df['track.id'].unique()[0]

    # print("Similarity between Track",trajA_id, "& Track",trajB_id,"using",
    # str(method),"method:")

    methods = ['pcm', 'frechet_dist', 'area_between_two_curves',
               'curve_length_measure', 'dtw']

    trajA_np = np.zeros((trajA.df.count()[1], 2))
    trajA_np[:, 0] = trajA.df['geometry'].x
    trajA_np[:, 1] = trajA.df['geometry'].y

    trajB_np = np.zeros((trajB.df.count()[1], 2))
    trajB_np[:, 0] = trajB.df['geometry'].x
    trajB_np[:, 1] = trajB.df['geometry'].y

    if(method not in methods):
        raise RuntimeError('Method not available')

    else:
        similarity_method = getattr(similaritymeasures, method)

        if(method == 'dtw'):
            sim, dtw_matrix = similarity_method(trajA_np, trajB_np)
            similarity = 1/(1+sim)
            return similarity
        else:
            similarity = 1/(1+similarity_method(trajA_np, trajB_np))
            return similarity


def crossed_similarity(list_traj, method):

    """ Compute similarity measures of a list of trajectories

        Keyword Arguments:
        list_traj {list}        -- List containing movingpandas trajectories
        method {string}         -- Name of the method to compute similarity
                                    pcm: Partial Curve Mapping
                                    frechet_dist: Discrete Frechet distance
                                    area_between_two_curves: Area method
                                    curve_length_measure: Curve Length
                                    dtw: Dynamic Time Warping

        Returns:

        df{dataframe}            -- Dataframe with summary of similarity
                                    measuresof all posible combinations from
                                    the trajectory list (list_traj)                                    
        """

    n = (len(list_traj))

    if(n <= 1):
        raise RuntimeError('More than 1 trajectory is required')

    number_comb = factorial(n)/(factorial(n-2)*factorial(2))

    start = timer()
    traj1_name = []
    traj2_name = []
    similarity = []
    i = 0

    for combo in combinations(list_traj, 2):
        traj1_name.append(combo[0].df['track.id'].unique()[0])
        traj2_name.append(combo[1].df['track.id'].unique()[0])
        simi = track_similarity(combo[0], combo[1], method)     
        similarity.append(simi)
        i += 1

        if (i % 10 == 0 or i == number_comb):
            print(round(i/number_comb*100, 1), "% of ", "calculations", sep='',
                  end='\r')

    df = pd.DataFrame(list(zip(traj1_name, traj2_name, similarity)),
                      columns=['Trajectory_1', 'Trajectory_2', 'Correlation'])
    df = df.sort_values(by=['Correlation'], ascending=False
                        ).reset_index(drop=True)

    end = timer()
    time = end-start

    print("\n%s similarity measures in %0.2f seconds" % (i, time))
    return(df)


def get_similarity_matrix(df):

    """ Returns a similarity matrix using the crossed similarity dataframe

        Keyword Arguments:
        df{df}         -- Crossed similarity dataframe

        Returns:

        df{dataframe}  -- Similarity matrix of trajectories (Symmetric matrix)

        """

    uniq_traj = np.unique(list(df['Trajectory_1'].unique())+list(
        df['Trajectory_2'].unique()))
    number_uniqtraj = len(uniq_traj)

    similarity_diagonal = [1] * number_uniqtraj
    df_diagonal = pd.DataFrame(list(zip(uniq_traj, uniq_traj,
                               similarity_diagonal)), columns=['Trajectory_1',
                               'Trajectory_2', 'Correlation'])
    frames = [df, df_diagonal]
    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values(by=['Correlation'], ascending=False).reset_index(
        drop=True)
    df = df.pivot(index='Trajectory_1', columns='Trajectory_2',
                  values='Correlation').copy()

    df = df.transpose().fillna(0)+df.fillna(0)
    df = df.replace(2, 1)

    return(df)


def plot_similarity_matrix(df_similarity_matrix, title):

    """ Generates similarity matrix plot

        Keyword Arguments:
        df{dataframe}      -- Similarity matrix of trajectories
        """

    sum_corr = list(df_similarity_matrix.sum().sort_values(
        ascending=True).index.values)
    df = df_similarity_matrix.sort_values(by=sum_corr).sort_index(
        axis=0, level=sum_corr)
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df, fignum=f.number)
    plt.title(title, y=1.2, fontsize=25)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
