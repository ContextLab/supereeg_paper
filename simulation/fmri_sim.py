import supereeg as se
import numpy as np
import sys
import os
import glob as glob
from scipy.spatial.distance import cdist
from config import config

bos =glob.glob(os.path.join(config['bo_datadir'], '*.bo'))

bo_locs = []
for i in bos:
    bo = se.load(i)
    bo_locs.append(bo.get_locs().as_matrix())


bo = se.load(bos[0])


std = se.Brain(se.load('std', vox_size=7))

d = cdist(bo.get_locs(), std.get_locs(), metric='Euclidean')

from supereeg.helpers import get_rows

for i in range(len(bo.get_locs())):
    min_ind = list(zip(*np.where(d == d.min())))[0]
    bo.locs.iloc[min_ind[0], :] = std.locs.iloc[min_ind[1], :]
    d[min_ind[0]] = np.inf
    d[:, min_ind[1]] = np.inf

sub_inds = get_rows(std.get_locs().values, bo.get_locs().values)

std[:, sub_inds]
