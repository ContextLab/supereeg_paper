import supereeg as se
from supereeg.helpers import _unique, known_unknown
import os
import pandas as pd
import numpy as np
import glob as glob
from config import config

results_dir = config['locs_resultsdir']


full_bos =glob.glob(os.path.join(config['bof_datadir'], '*.bo'))

union_locs = []
bo_locs = []
for i in full_bos:
    try:
        bo = se.load(i)
        if union_locs == []:
            union_locs = bo.get_locs().as_matrix()
        else:
            union_locs = np.vstack((union_locs, bo.get_locs().as_matrix()))

    except:
        pass

full_locations, l_indices = _unique(union_locs)

sub_bos =glob.glob(os.path.join(config['bof_sub_datadir'], '*.bo'))

union_locs = []
bo_locs = []
for i in sub_bos:
    try:
        bo = se.load(i)
        if union_locs == []:
            union_locs = bo.get_locs().as_matrix()
        else:
            union_locs = np.vstack((union_locs, bo.get_locs().as_matrix()))

    except:
        pass

union_locations, l_indices = _unique(union_locs)

known_inds, unknown_inds = known_unknown(full_locations, union_locations, union_locations)

### if you want to downsample:
sub_inds = unknown_inds[::100]



results = os.path.join(results_dir, 'mo_locs.npz')

np.savez(results, full_locs = full_locations, union_locs = union_locations,
         sub_locs = full_locations[sub_inds])
