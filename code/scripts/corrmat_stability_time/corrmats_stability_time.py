
import supereeg as se
import numpy as np
import sys
import os
import matplotlib.ticker as tkr
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from config import config


fname = sys.argv[1]

data = sys.argv[2]

chunk_size = 1000

rand_iters = 2

results_dir = os.path.join(config['resultsdir'], data)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


def electrode_search(fname, threshold=10):
    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    return sum(~thresh_bool)


def corr_corrmats(bo_1, bo_2):

    mo_1 = se.Model(bo_1)
    mo_2 = se.Model(bo_2)

    tril_1 = np.atleast_2d(mo_1.get_model()[np.tril_indices(locs_total, k=-1)])
    tril_2 = np.atleast_2d(mo_2.get_model()[np.tril_indices(locs_total, k=-1)])

    return 1 - cdist(tril_1, tril_2, 'correlation')


print(fname)

elec_count = electrode_search(sys.argv[1])

if elec_count > 1:

    fname =os.path.basename(os.path.splitext(fname)[0])

    print('running time stability for : ' + fname)

    # load brain object
    bo = se.load(sys.argv[1])

    # filter
    bo.apply_filter()

    time_total = bo.get_data().shape[0]
    locs_total = bo.get_data().shape[1]

    divided_time = int(time_total/ 2)

    chunk_totals = int(divided_time / chunk_size)

    full_data = bo.get_data().values

    all_idx = np.arange(time_total)
    half_1_idx = np.random.choice(time_total, replace=False, size=divided_time)
    half_2_idx = np.delete(all_idx, half_1_idx)

    n_samples = chunk_totals

    corrs_rand = np.zeros((rand_iters, n_samples))

    half_1_bo = bo[half_1_idx]
    half_2_bo = bo[half_2_idx]


    for r in np.arange(rand_iters):

        unused = np.arange(divided_time)
        current = np.array([])

        for i in np.arange(n_samples):

            current_idx = np.random.choice(np.arange(unused.shape[0]), replace=False, size=chunk_size)
            if current.shape[0] == 0:
                current = unused[current_idx]
            else:

                current = np.concatenate((current, unused[current_idx]))
            unused_idx = np.delete(np.arange(unused.shape[0]), current_idx)
            unused = unused[unused_idx]
            corrs_rand[r, i] = corr_corrmats(half_1_bo, half_2_bo[current])[0][0]

    corrs_rand = np.mean(corrs_rand, axis=0)

    parse_close_1 = divided_time
    parse_close_2 = divided_time
    parse_apart_1 = 0
    parse_apart_2 = time_total

    n_samples = chunk_totals

    corrs_close = np.zeros((2, n_samples))
    corrs_apart = np.zeros((2, n_samples))

    for i in np.arange(n_samples):

        for t in np.arange(2):

            if t == 0:
                parse_close_2 += chunk_size
                parse_apart_2 -= chunk_size

                corrs_close[t, i] = corr_corrmats(bo[:divided_time], bo[divided_time:parse_close_2])[0][0]
                corrs_apart[t, i] = corr_corrmats(bo[:divided_time], bo[parse_apart_2:time_total])[0][0]

                print(parse_close_2)
                print(parse_apart_2)
            else:
                parse_close_1 -= chunk_size
                parse_apart_1 += chunk_size

                corrs_close[t, i] = corr_corrmats(bo[divided_time:], bo[parse_close_1:divided_time])[0][0]
                corrs_apart[t, i] = corr_corrmats(bo[divided_time:], bo[0:parse_apart_1])[0][0]

                print(parse_close_1)
                print(parse_apart_1)

    corrs_apart = np.mean(corrs_apart, axis=0)
    corrs_close = np.mean(corrs_close, axis=0)

    stable_npz = os.path.join(results_dir, fname + '.npz')

    np.savez(stable_npz, rand=corrs_rand, apart=corrs_apart, close=corrs_close, sample_rate=bo.sample_rate[0],
             samples=time_total, chunk_size=chunk_size)




else:
    print('skipping bo (not enough electrodes pass kurtosis threshold): ' + sys.argv[1])