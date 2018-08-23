import supereeg as se
import numpy as np
import sys
import os
import glob as glob
from config import config



fname = sys.argv[1]

results_dir = os.path.join(config['model_datadir'])

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

locs_dir = config['locs_resultsdir']

bo_dir = config['bof_datadir']

locs_data = np.load(os.path.join(locs_dir, 'locs.npz'))

locs = locs_data['locs']


fname = os.path.basename(os.path.splitext(fname)[0])

print('creating model object: ' + fname)

# load brain object
bo = se.load(sys.argv[1])

# filter
bo.filter=None

# make model
mo = se.Model(bo, locs=locs)

# save model
mo.save(os.path.join(results_dir, fname))
