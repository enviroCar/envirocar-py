# -*- coding: utf-8 -*-

from copy import copy, deepcopy
from shapely.geometry import LineString

from movingpandas import Trajectory, TrajectoryCollection
from .geom_utils import measure_distance_spherical, measure_distance_euclidean


class TrackGeneralizer:
    """
    Generalizer base class
    """

    def __init__(self, traj):
        """
        Create TrajectoryGeneralizer
        Parameters
        ----------
        traj : Trajectory/TrajectoryCollection
        """
        self.traj = traj

    def generalize(self, tolerance, columnNamesToDistributeValues):
        """
        Generalize the input Trajectory/TrajectoryCollection.
        Parameters
        ----------
        tolerance : any type
            Tolerance threshold, differs by generalizer
        columnNamesToDistributeValues: list
            List of column names to distribute values to neighboring kept rows
        Returns
        -------
        Trajectory/TrajectoryCollection
            Generalized Trajectory or TrajectoryCollection
        """
        if isinstance(self.traj, Trajectory):
            return self._generalize_traj(
                self.traj, tolerance, columnNamesToDistributeValues)
        elif isinstance(self.traj, TrajectoryCollection):
            return self._generalize_traj_collection(
                tolerance, columnNamesToDistributeValues)
        else:
            raise TypeError

    def _generalize_traj_collection(self, tolerance,
                                    columnNamesToDistributeValues):
        generalized = []
        for traj in self.traj.trajectories:
            generalized.append(self._generalize_traj(
                traj, tolerance, columnNamesToDistributeValues))
        result = copy(self.traj)
        result.trajectories = generalized
        return result

    def _generalize_traj(self, traj, tolerance, columnNamesToDistributeValues):
        return traj


class MinDistanceGeneralizer(TrackGeneralizer):
    """
    Generalizes based on distance.
    This generalization ensures that consecutive locations are at least a
        certain distance apart.

    tolerance : float
        Desired minimum distance between consecutive points
    columnNamesToDistributeValues : list of column names to distribute values
        to neighboring kept rows
    Examples
    --------
    >>> mpd.MinDistanceGeneralizer(traj).generalize(tolerance=1.0)
    """

    def _generalize_traj(self, traj, tolerance,
                         columnNamesToDistributeValues=None):
        temp_df = traj.df.copy()
        prev_pt = temp_df.iloc[0]['geometry']
        keep_rows = [0]
        i = 0
        trajCopy = deepcopy(traj)
        for index, row in temp_df.iterrows():
            pt = row['geometry']
            if traj.is_latlon:
                dist = measure_distance_spherical(pt, prev_pt)
            else:
                dist = measure_distance_euclidean(pt, prev_pt)
            if dist >= tolerance:
                keep_rows.append(i)
                prev_pt = pt
            i += 1

        keep_rows.append(len(traj.df)-1)

        if (columnNamesToDistributeValues):
            # Distribute the selected values of dropped rows to the
            # neighboring rows
            for i, rowIndex in enumerate(keep_rows):
                if (i < len(keep_rows) - 1 and keep_rows[i+1] - rowIndex > 1):
                    nextRowIndex = keep_rows[i + 1]
                    discardedRows = trajCopy.df.iloc[rowIndex +
                                                     1: nextRowIndex]
                    discardedRowsSelectedColumns = discardedRows[
                        columnNamesToDistributeValues]
                    discardedRowsSelectedColumnsSum = \
                        discardedRowsSelectedColumns.sum()
                    aboveRow = trajCopy.df.iloc[rowIndex]
                    belowRow = trajCopy.df.iloc[nextRowIndex]
                    aboveRow[columnNamesToDistributeValues] = \
                        aboveRow[columnNamesToDistributeValues] + (
                        discardedRowsSelectedColumnsSum/2)
                    belowRow[columnNamesToDistributeValues] = \
                        belowRow[columnNamesToDistributeValues] + (
                        discardedRowsSelectedColumnsSum/2)
                    trajCopy.df.iloc[rowIndex] = aboveRow
                    trajCopy.df.iloc[nextRowIndex] = belowRow

        new_df = trajCopy.df.iloc[keep_rows]
        removedRowsCount = len(traj.df.index) - len(keep_rows)
        new_columns = {'Generalized': True, 'Generalization.Method': 'Min-Distance',
                       'Generalization.RemovedRowsCount': removedRowsCount}
        new_df = new_df.assign(**new_columns)
        new_traj = Trajectory(new_df, trajCopy.id)
        return new_traj


class MinTimeDeltaGeneralizer(TrackGeneralizer):
    """
    Generalizes based on time.
    This generalization ensures that consecutive rows are at least a certain
        timedelta apart.
    tolerance : datetime.timedelta
        Desired minimum time difference between consecutive rows
    columnNamesToDistributeValues : list of column names to distribute values
        to neighboring kept rows
    Examples
    --------
    >>> mpd.MinTimeDeltaGeneralizer(traj).generalize(tolerance=timedelta(
            minutes=10))
    """

    def _generalize_traj(self, traj, tolerance,
                         columnNamesToDistributeValues=None):
        temp_df = traj.df.copy()
        temp_df['t'] = temp_df.index
        prev_t = temp_df.head(1)['t'][0]
        keep_rows = [0]
        i = 0
        trajCopy = deepcopy(traj)
        for index, row in temp_df.iterrows():
            t = row['t']
            tdiff = t - prev_t
            if tdiff >= tolerance:
                keep_rows.append(i)
                prev_t = t
            i += 1

        keep_rows.append(len(traj.df)-1)

        if (columnNamesToDistributeValues):
            # Distribute the selected values of dropped rows to the
            # neighboring rows
            for i, rowIndex in enumerate(keep_rows):
                if (i < len(keep_rows) - 1 and keep_rows[i+1] - rowIndex > 1):
                    nextRowIndex = keep_rows[i + 1]
                    discardedRows = trajCopy.df.iloc[rowIndex +
                                                     1: nextRowIndex]
                    discardedRowsSelectedColumns = \
                        discardedRows[columnNamesToDistributeValues]
                    discardedRowsSelectedColumnsSum = \
                        discardedRowsSelectedColumns.sum()
                    aboveRow = trajCopy.df.iloc[rowIndex]
                    belowRow = trajCopy.df.iloc[nextRowIndex]
                    aboveRow[columnNamesToDistributeValues] = \
                        aboveRow[columnNamesToDistributeValues] + (
                        discardedRowsSelectedColumnsSum/2)
                    belowRow[columnNamesToDistributeValues] = \
                        belowRow[columnNamesToDistributeValues] + (
                        discardedRowsSelectedColumnsSum/2)
                    trajCopy.df.iloc[rowIndex] = aboveRow
                    trajCopy.df.iloc[nextRowIndex] = belowRow

        new_df = trajCopy.df.iloc[keep_rows]
        removedRowsCount = len(traj.df.index) - len(keep_rows)
        new_columns = {'Generalized': True, 'Generalization.Method': 'Min-Time-Delta',
                       'Generalization.RemovedRowsCount': removedRowsCount}
        new_df = new_df.assign(**new_columns)
        new_traj = Trajectory(new_df, trajCopy.id)
        return new_traj


class DouglasPeuckerGeneralizer(TrackGeneralizer):
    """
    Generalizes using Douglas-Peucker algorithm.
    tolerance : float
        Distance tolerance
    columnNamesToDistributeValues : list of column names to distribute values
        to neighboring kept rows
    Examples
    --------
    >>> mpd.DouglasPeuckerGeneralizer(traj).generalize(tolerance=1.0)
    """

    def _generalize_traj(self, traj, tolerance,
                         columnNamesToDistributeValues=None):
        prev_pt = None
        pts = []
        keep_rows = []
        i = 0
        trajCopy = deepcopy(traj)
        for index, row in trajCopy.df.iterrows():
            current_pt = row.geometry
            # Handle first row and skip the loop
            if prev_pt is None:
                prev_pt = current_pt
                keep_rows.append(i)
                continue
            line = LineString([prev_pt, current_pt])
            for pt in pts:
                if line.distance(pt) > tolerance:
                    prev_pt = current_pt
                    pts = []
                    keep_rows.append(i)
                    continue
            pts.append(current_pt)
            i += 1
        # Keep the last row
        keep_rows.append(i)

        if (columnNamesToDistributeValues):
            # Distribute the selected values of dropped rows to the
            # neighboring rows
            for i, rowIndex in enumerate(keep_rows):
                if (i < len(keep_rows) - 1 and keep_rows[i+1] - rowIndex > 1):
                    nextRowIndex = keep_rows[i + 1]
                    discardedRows = trajCopy.df.iloc[rowIndex +
                                                     1: nextRowIndex]
                    discardedRowsSelectedColumns = \
                        discardedRows[columnNamesToDistributeValues]
                    discardedRowsSelectedColumnsSum = \
                        discardedRowsSelectedColumns.sum()
                    aboveRow = trajCopy.df.iloc[rowIndex]
                    belowRow = trajCopy.df.iloc[nextRowIndex]
                    aboveRow[columnNamesToDistributeValues] = \
                        aboveRow[columnNamesToDistributeValues] + (
                        discardedRowsSelectedColumnsSum/2)
                    belowRow[columnNamesToDistributeValues] = \
                        belowRow[columnNamesToDistributeValues] + (
                        discardedRowsSelectedColumnsSum/2)
                    trajCopy.df.iloc[rowIndex] = aboveRow
                    trajCopy.df.iloc[nextRowIndex] = belowRow

        new_df = trajCopy.df.iloc[keep_rows]
        removedRowsCount = len(traj.df.index) - len(keep_rows)
        new_columns = {'Generalized': True, 'Generalization.Method': 'Douglas-Peucker',
                       'Generalization.RemovedRowsCount': removedRowsCount}
        new_df = new_df.assign(**new_columns)
        new_traj = Trajectory(new_df, trajCopy.id)
        return new_traj
