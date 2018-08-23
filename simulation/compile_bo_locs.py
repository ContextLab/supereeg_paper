import supereeg as se
from supereeg.helpers import _unique
import os
import pandas as pd
import numpy as np
import glob as glob
from config import config

results_dir = config['locs_resultsdir']

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


def electrode_search(fname, threshold=10):

    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    locs = se.load(fname, field='locs')
    if sum(~thresh_bool) > 1:
        locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])
        return locs[~thresh_bool]


bos =glob.glob(os.path.join(config['bo_datadir'], '*.bo'))

union_locs = []
bo_locs = []
for i in bos:
    try:
        locs = electrode_search(i)
        if not locs.empty:
            bo = se.load(i)
            bo_locs.append(bo.get_locs().as_matrix())
            if union_locs == []:
                union_locs = locs.as_matrix()
            else:
                union_locs = np.vstack((union_locs, locs.as_matrix()))

    except:
        pass

locations, l_indices = _unique(union_locs)

results = os.path.join(results_dir, 'locs.npz')

np.savez(results, locs = locations, loc_list = bo_locs)
