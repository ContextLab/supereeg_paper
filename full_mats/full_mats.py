
import supereeg as se
from supereeg.model import _bo2model
from scipy.spatial.distance import cdist, pdist, squareform
from supereeg.helpers import _r2z, _z2r, _get_corrmat
import numpy as np
import glob
import sys
import multiprocessing
import pandas as pd
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from config import config
from joblib import Parallel, delayed


fname = sys.argv[1]

model_template = sys.argv[2]

radius = sys.argv[3]

results_dir = os.path.join(config['resultsdir'], model_template +"_"+ radius)

results_log_dir = os.path.join(config['resultsdir'], model_template +"_"+ radius+"_log")

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

try:
    if not os.path.exists(results_log_dir):
        os.makedirs(results_log_dir)
except OSError as err:
   print(err)


def electrode_search(fname, threshold=10):
    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    return sum(~thresh_bool)

def _rbf(x, center, width=20):
    return np.exp(-cdist(x, center, metric='euclidean') ** 2 / float(width))


def _compute_coord(coord, weights, Z):
    next_weights = np.outer(weights[coord[0], :], weights[coord[1], :])
    next_weights = next_weights - np.triu(next_weights)
    return np.sum(next_weights), np.sum(Z * next_weights)

def _expand_corrmat_predict(C, weights):
    """
    Gets full correlation matrix
    Parameters
    ----------
    C : Numpy array
        Subject's correlation matrix
    weights : Numpy array
        Weights matrix calculated using _rbf function matrix
    mode : str
        Specifies whether to compute over all elecs (fit mode) or just new elecs
        (predict mode)
    Returns
    ----------
    numerator : Numpy array
        Numerator for the expanded correlation matrix
    denominator : Numpy array
        Denominator for the expanded correlation matrix
    """

    C[np.eye(C.shape[0]) == 1] = 0
    C[np.where(np.isnan(C))] = 0

    n = weights.shape[0]
    K = np.zeros([n, n])
    W = np.zeros([n, n])
    Z = C

    s = C.shape[0]
    sliced_up = [(x, y) for x in range(s, n) for y in range(x)]

    results = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(_compute_coord)(coord, weights, Z) for coord in sliced_up)

    W[[x[0] for x in sliced_up], [x[1] for x in sliced_up]] = [x[0] for x in results]
    K[[x[0] for x in sliced_up], [x[1] for x in sliced_up]] = [x[1] for x in results]

    return (K + K.T), (W + W.T)

def _bo2model_old(bo, locs, width=20):
    """Returns numerator and denominator given a brain object"""
    sub_corrmat = _get_corrmat(bo)
    np.fill_diagonal(sub_corrmat, 0)
    sub_corrmat_z = _r2z(sub_corrmat)
    sub_rbf_weights = _rbf(locs, bo.get_locs(), width=width)
    n, d = _expand_corrmat_predict(sub_corrmat_z, sub_rbf_weights)

    return n, d, 1

locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')

R = np.load(locs_file)['locs']

elec_count = electrode_search(sys.argv[1])

if elec_count > 1:

    fname =os.path.basename(os.path.splitext(fname)[0])

    print('creating model object: ' + fname)

    # load brain object
    bo = se.load(sys.argv[1])

    # filter
    bo = bo.get_filtered_bo()

    npz_outfile = sys.argv[1][:-2]+'npz'
    if not os.path.isfile(npz_outfile):
        np.savez(npz_outfile, Y = bo.get_data().as_matrix(),
                          R = bo.get_locs().as_matrix(),
                          fname_labels=np.atleast_2d(bo.sessions.as_matrix()),
                          sample_rate=bo.sample_rate)


    # exapand matrix
    num_corrmat_x, denom_corrmat_x, n_subs = _bo2model(bo, R, int(radius))

    mo = se.Model(bo, locs=R)

    #### save the z expanded correlation matrix
    print('saving model object: ' + fname)

    np.savez(os.path.join(results_log_dir, fname), num = num_corrmat_x, den = denom_corrmat_x)

    mo.save(os.path.join(results_log_dir, fname))
    # exapand matrix
    num_corrmat_x, denom_corrmat_x, n_subs = _bo2model_old(bo, R, int(radius))

    #### save the z expanded correlation matrix
    print('saving model object: ' + fname)
    np.savez(os.path.join(results_dir, fname), num = num_corrmat_x, den = denom_corrmat_x)
else:
    print('skipping model (not enough electrodes pass kurtosis threshold): ' + sys.argv[1])