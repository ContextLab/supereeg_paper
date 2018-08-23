import supereeg as se
import numpy as np
import sys
import os
import glob as glob
from config import config


def electrode_search(fname, threshold=10):
    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    return sum(~thresh_bool)


fname = sys.argv[1]

results_dir = os.path.join(config['model_dir'])

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

locs_dir = config['locs_resultsdir']

bo_dir = config['bof_datadir']

locs_data = np.load(os.path.join(locs_dir, 'locs.npz'))

locs = locs_data['locs']

elec_count = electrode_search(sys.argv[1])

if elec_count > 1:

    fname = os.path.basename(os.path.splitext(fname)[0])

    print('creating model object: ' + fname)

    # load brain object
    bo = se.load(sys.argv[1])

    # filter
    bo.apply_filter()

    # make model
    mo = se.Model(bo, locs=locs)

    # save model
    mo.save(os.path.join(results_dir, fname))

else:
    print('skipping model (not enough electrodes pass kurtosis threshold): ' + sys.argv[1])