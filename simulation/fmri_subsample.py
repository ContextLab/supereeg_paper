import supereeg as se
import numpy as np
from supereeg.helpers import get_rows
import sys
import os
import glob as glob
from scipy.spatial.distance import cdist
from config import config


locs_dir = config['locs_resultsdir']

fmri_dir = config['fmri_datadir']

bo_dir = config['bof_datadir']

results_dir = os.path.join(config['bof_sub_datadir'])

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

locs_data = np.load(os.path.join(locs_dir, 'locs.npz'))

loc_list = locs_data['loc_list']

locs = locs_data['locs']


subs =list(range(1, len(glob.glob(os.path.join(bo_dir, '*.bo'))) + 1))

for i, loc in enumerate(loc_list):

    ind = np.random.choice(subs, 1)
    bo = se.load(os.path.join(bo_dir, 'sub-%d.bo' % ind))

    d = cdist(loc, bo.get_locs().values, metric='Euclidean')

    for l in range(len(loc)):
        min_ind = list(zip(*np.where(d == d.min())))[0]
        loc[min_ind[0], :] = bo.get_locs().values[min_ind[1], :]
        d[min_ind[0]] = np.inf
        d[:, min_ind[1]] = np.inf

    sub_inds = get_rows(bo.get_locs().values, loc)

    subbo = bo[:, sub_inds]

    subbo.save(os.path.join(results_dir, 'fmri_subsampled_s_%d_%d.bo' % (i, ind)))


