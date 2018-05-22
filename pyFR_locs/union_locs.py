import supereeg as se
import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from config import config
from nilearn import plotting as ni_plt

## this script iterates over brain objects, filters them based on kurtosis value,
## then compiles the clean electrodes into a numpy array as well as a list of the contributing brain objects


results_dir = config['resultsdir']

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


data_dir = config['datadir']


def electrode_search(fname, threshold=10):

    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    locs = se.load(fname, field='locs')
    if sum(~thresh_bool) > 1:
        locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])
    return locs[~thresh_bool]

union_locs = []
model_data = []

brain_data = glob.glob(os.path.join(data_dir, '*.bo'))
for i in brain_data:
    try:
        locs = electrode_search(i)
        if not locs.empty:
            if union_locs == []:
                print(os.path.basename(i))
                union_locs = locs.as_matrix()
                model_data.append(os.path.basename(i))
            else:
                union_locs = np.vstack((union_locs, locs.as_matrix()))
                model_data.append(os.path.basename(i))
    except:
        pass

locations = se.sort_unique_locs(union_locs)

results = os.path.join(results_dir, 'locs.npz')
#results = os.path.join('/scratch/lucy.owen/supereeg/', 'locs.npz')

np.savez(results, locs = locations, subjs = model_data)