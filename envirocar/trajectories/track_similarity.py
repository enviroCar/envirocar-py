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


def trajCollections_similarity(trajCollectionA, trajCollectionB, method):
    """ Compute similarity measures using the similaritymeasures package
        https://pypi.org/project/similaritymeasures/

        Keyword Arguments:
        trajCollectionA {movingpandas trajectory}  -- movingpandas trajectory
        trajCollectionB {movingpandas trajectory}  -- movingpandas trajectory
        method {string}                            -- Name of the method to compute
                                                      similarity
                                                        pcm: Partial Curve Mapping
                                                        frechet_dist: Discrete Frechet distance
                                                        area_between_two_curves: Area method
                                                        curve_length_measure: Curve Length
                                                        dtw: Dynamic Time Warping

        Returns:
        similarity dataframe                       -- Float value (0,1) corresponding to the computed similarity. 
        """

    n = len(trajCollectionA.trajectories)
    m = len(trajCollectionB.trajectories)

    if(n != m):
        raise RuntimeError('Trajectory collections should be the same size !')

    traj1_name = []
    traj2_name = []
    similarity = []

    for i in range(n):

        traj_a = trajCollectionA.trajectories[i]
        traj_b = trajCollectionB.trajectories[i]

        traj1_name.append(traj_a.df['track.id'].unique()[0])
        traj2_name.append(traj_b.df['track.id'].unique()[0])

        simi = track_similarity(traj_a, traj_b, method)
        similarity.append(simi)

    traj_a_gen = 'Generalized' in traj_a.df.columns
    traj_b_gen = 'Generalized' in traj_b.df.columns

    column_name_1 = "Trajectory_1"
    column_name_2 = "Trajectory_2"

    if traj_a_gen == True:
        column_name_1 = "Generalized_"+column_name_1
    if traj_b_gen == True:
        column_name_2 = "Generalized_"+column_name_2

    df = pd.DataFrame(list(zip(traj1_name, traj2_name, similarity)),
                      columns=[column_name_1, column_name_2, 'Similarity'])
    return(df)


def crossed_similarity(trajCollection, method):
    """ Compute similarity measures of a list of trajectories

        Keyword Arguments:
        trajCollection {trajectoryCollection}        -- List containing movingpandas trajectories
        method {string}                              -- Name of the method to compute similarity
                                                        pcm: Partial Curve Mapping
                                                        frechet_dist: Discrete Frechet distance
                                                        area_between_two_curves: Area method
                                                        curve_length_measure: Curve Length
                                                        dtw: Dynamic Time Warping
        Returns:

        df{dataframe}                                -- Dataframe with summary of similarity 
                                                        measures of all posible combinations from
                                                        the trajectory list (list_traj)

        """

    n = (len(trajCollection.trajectories))

    if(n <= 1):
        raise RuntimeError('More than 1 trajectory is required')

    trajVector = []
    for i in (trajCollection.trajectories):
        trajVector.append(i)

    number_comb = factorial(n)/(factorial(n-2)*factorial(2))

    start = timer()
    traj1_name = []
    traj2_name = []
    similarity = []
    i = 0

    for combo in combinations(trajVector, 2):
        traj1_name.append(combo[0].df['track.id'].unique()[0])
        traj2_name.append(combo[1].df['track.id'].unique()[0])
        simi = track_similarity(combo[0], combo[1], method)
        similarity.append(simi)
        i += 1

        if (i % 10 == 0 or i == number_comb):
            print(round(i/number_comb*100, 1), "% of ", "calculations", sep='',
                  end='\r')

    df = pd.DataFrame(list(zip(traj1_name, traj2_name, similarity)),
                      columns=['Trajectory_1', 'Trajectory_2', 'Similarity'])

    df_2 = pd.DataFrame(list(zip(traj2_name, traj1_name, similarity)),
                        columns=['Trajectory_1', 'Trajectory_2', 'Similarity'])

    frames = [df, df_2]

    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values(by=['Similarity'], ascending=False
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
                                                                        'Trajectory_2', 'Similarity'])
    frames = [df, df_diagonal]
    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values(by=['Similarity'], ascending=False).reset_index(
        drop=True)

    df = df.pivot(index='Trajectory_1', columns='Trajectory_2',
                  values='Similarity').copy()

    return(df)


def plot_similarity_matrix(df_similarity_matrix, title):
    """ Generates similarity matrix plot

        Keyword Arguments:
        df{dataframe}      -- Similarity matrix of trajectories
        """

    sum_corr = list(df_similarity_matrix.sum(
    ).sort_values(ascending=True).index.values)
    df_similarity_matrix = df_similarity_matrix[sum_corr]
    df = df_similarity_matrix.reindex(sum_corr)

    f = plt.figure(figsize=(19, 15))
    plt.matshow(df, fignum=f.number)
    plt.title(title, y=1.2, fontsize=25)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
