
import supereeg as se
import numpy as np
import sys
import os
import glob as glob
import matplotlib.ticker as tkr
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from config import config


data = sys.argv[1]


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


def corr_corrmats(mo_1, mo_2):

    locs_total = mo_1.get_model().shape[1]

    tril_1 = np.atleast_2d(mo_1.get_model()[np.tril_indices(locs_total, k=-1)])
    tril_2 = np.atleast_2d(mo_2.get_model()[np.tril_indices(locs_total, k=-1)])

    return 1 - cdist(tril_1, tril_2, 'correlation')



print('running time stability across patients for: ' + data)

files = glob.glob(os.path.join(config[data + 'datadir'], '*.mo'))


total_patients = len(files)

divided_patients = int(total_patients/ 2)

all_idx = np.arange(total_patients)
half_1_idx = np.random.choice(total_patients, replace=False, size=divided_patients)
half_2_idx = np.delete(all_idx, half_1_idx)

np.random.shuffle(half_2_idx)

n_samples = len(half_2_idx)

corrs_rand = np.zeros((rand_iters, n_samples))


half_1_mo = se.Model(list(np.array(files)[half_1_idx]), n_subs=len(half_1_idx))


for r in np.arange(rand_iters):

    for e, i in enumerate(half_2_idx):

        if e == 0:

            mo_2 = se.Model(np.array(files)[half_2_idx[e]], n_subs=1)

        else:

            mo_i = se.Model(np.array(files)[half_2_idx[e]], n_subs=1)

            mo_2 = mo_2 + mo_i

        corrs_rand[r, e] = corr_corrmats(half_1_mo, mo_2)[0][0]

#corrs_rand = np.mean(corrs_rand, axis=0)

stable_npz = os.path.join(results_dir, data + '.npz')

np.savez(stable_npz, rand=corrs_rand)


