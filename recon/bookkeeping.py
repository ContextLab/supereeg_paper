import numpy as np
from glob import glob as lsdir
from stats import round_it, r2z, z2r
import os
from scipy.spatial.distance import squareform

def get_rows(all_locations, subj_locations):
    """
        This function indexes a subject's electrode locations in the full array of electrode locations

        Parameters
        ----------
        all_locations : ndarray
            Full array of electrode locations

        subj_locations : ndarray
            Array of subject's electrode locations

        Returns
        ----------
        results : list
            Indexs for subject electrodes in the full array of electrodes

        """
    if subj_locations.ndim == 1:
        subj_locations = subj_locations.reshape(1, 3)
    inds = np.full([1, subj_locations.shape[0]], np.nan)
    for i in range(subj_locations.shape[0]):
        possible_locations = np.ones([all_locations.shape[0], 1])
        try:
            for c in range(all_locations.shape[1]):
                possible_locations[all_locations[:, c] != subj_locations[i, c], :] = 0
            inds[0, i] = np.where(possible_locations == 1)[0][0]
        except:
            pass
    inds = inds[~np.isnan(inds)]
    return [int(x) for x in inds]


def known_unknown(fullarray, knownarray, subarray=None, electrode=None):
    """
        This finds the indices for known and unknown electrodes in the full array of electrode locations

        Parameters
        ----------
        fullarray : ndarray
            Full array of electrode locations - All electrodes that pass the kurtosis test

        knownarray : ndarray
            Subset of known electrode locations  - Subject's electrode locations that pass the kurtosis test (in the leave one out case, this is also has the specified location missing)

        subarray : ndarray
            Subject's electrode locations (all)

        electrode : str
            Index of electrode in subarray to remove (in the leave one out case)

        Returns
        ----------
        known_inds : list
            List of known indices

        unknown_inds : list
            List of unknown indices

        """
    ## where known electrodes are located in full matrix
    known_inds = get_rows(round_it(fullarray, 3), round_it(knownarray, 3))
    ## where the rest of the electrodes are located
    unknown_inds = list(set(range(np.shape(fullarray)[0])) - set(known_inds))
    if not electrode is None:
        ## where the removed electrode is located in full matrix
        rm_full_ind = get_rows(round_it(fullarray, 3), round_it(subarray[int(electrode)], 3))
        ## where the removed electrode is located in the unknown index subset
        rm_unknown_ind = np.where(np.array(unknown_inds) == np.array(rm_full_ind))[0].tolist()
        return known_inds, unknown_inds, rm_unknown_ind
    else:
        return known_inds, unknown_inds

def remove_electrode(subkarray, subarray, electrode):
    """
        Removes electrode from larger array

        Parameters
        ----------
        subkarray : ndarray
            Subject's electrode locations that pass the kurtosis test

        subarray : ndarray
            Subject's electrode locations (all)

        electrode : str
            Index of electrode in subarray to remove

        Returns
        ----------
        results : ndarray
            Subject's electrode locations that pass kurtosis test with electrode removed

        """
    rm_ind = get_rows(subkarray, subarray[electrode])
    other_inds = [i for i in range(np.shape(subkarray)[0]) if i != electrode]
    return np.delete(subkarray, rm_ind, 0), other_inds


def logdiffexp(a, b):
    return np.add(a, np.log(np.subtract(1, np.exp(np.subtract(b, a)))))

def alter_avemat_1(Average_matrix, Subj_matrix):
    """
        Removes one subject's full correlation matrix from the average correlation matrix

        Parameters
        ----------
        Average_matrix : npz file
            npz file contains the fields:
                average_matrix : the average full correlation matrix for all subjects (n)
                n : number of full correlation matrices that contributed to average matrix

        Subj_matrix : list
            Subject's squareformed full correlation matrix

        Returns
        ----------
        results : ndarray
            Average matrix with one subject's data removed

        """

    ### this is the more correct way, but it decreases the reconstruction accruacy by staying it r space
    summed_matrix = Average_matrix['average_matrix'] * Average_matrix['n']
    Z_all = r2z(Average_matrix['average_matrix'])
    n = Average_matrix['n']
    summed_matrix = Z_all * n
    count_removed = n - 1
    C_est = np.divide(Subj_matrix['num'], Subj_matrix['den'])
    C_est[np.where(np.isnan(C_est))] = 0
    C_est = C_est + np.eye(C_est.shape[0])

    return z2r(np.divide(np.subtract(np.multiply(n, Z_all), C_est), n-1)), n-1


def alter_avemat_2(Average_matrix, Subj_matrix):
    """
        Removes one subject's full correlation matrix from the average correlation matrix

        Parameters
        ----------
        Average_matrix : npz file
            npz file contains the fields:
                average_matrix : the average full correlation matrix for all subjects (n)
                n : number of full correlation matrices that contributed to average matrix

        Subj_matrix : list
            Subject's squareformed full correlation matrix

        Returns
        ----------
        results : ndarray
            Average matrix with one subject's data removed

        """

    ### this is the more correct way, but it decreases the reconstruction accruacy by staying it r space
    summed_matrix = Average_matrix['average_matrix'] * Average_matrix['n']
    n = Average_matrix['n']
    Z_all = r2z(Average_matrix['average_matrix']* n)
    C_est = np.divide(Subj_matrix['num'], Subj_matrix['den'])
    C_est[np.where(np.isnan(C_est))] = 0
    C_est = C_est + np.eye(C_est.shape[0])

    return z2r(np.divide(np.subtract(Z_all, C_est), n-1)), n-1

# def alter_avemat(Average_matrix, Subj_matrix):
#     """
#         Removes one subject's full correlation matrix from the average correlation matrix
#
#         Parameters
#         ----------
#         Average_matrix : npz file
#             npz file contains the fields:
#                 average_matrix : the average full correlation matrix for all subjects (n)
#                 n : number of full correlation matrices that contributed to average matrix
#
#         Subj_matrix : list
#             Subject's squareformed full correlation matrix
#
#         Returns
#         ----------
#         results : ndarray
#             Average matrix with one subject's data removed
#
#         """
#     ### this is the incorrect way, but produces the same results as the paper
#     # summed_matrix = Average_matrix['average_matrix'] * Average_matrix['n']
#     # #Z_all = r2z(Average_matrix['average_matrix'])
#     # n = Average_matrix['n']
#     # #summed_matrix = Z_all * n
#     # count_removed = n - 1
#     # C_est = z2r(Subj_matrix['C_est'])
#     # C_est[np.where(np.isnan(C_est))] = 0
#     # #C_est = C_est + np.eye(C_est.shape[0])
#     # return (summed_matrix - (C_est + np.eye(C_est.shape[0])))/count_removed, count_removed
#
#     ### this is the more correct way, but it decreases the reconstruction accruacy by staying it r space
#     summed_matrix = Average_matrix['average_matrix'] * Average_matrix['n']
#     Z_all = r2z(Average_matrix['average_matrix'])
#     n = Average_matrix['n']
#     summed_matrix = Z_all * n
#     count_removed = n - 1
#     C_est = Subj_matrix['C_est']
#     C_est[np.where(np.isnan(C_est))] = 0
#     C_est = C_est + np.eye(C_est.shape[0])
#
#     return z2r(np.divide(np.subtract(np.multiply(n, Z_all), C_est), n-1)), n-1