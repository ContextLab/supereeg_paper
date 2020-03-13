import supereeg as se
from supereeg.helpers import _unique
import numpy as np
import glob
import os
import pandas as pd
from config import config
import sys

## this script iterates over brain objects, filters them based on kurtosis value,
## then compiles the clean electrodes into a numpy array as well as a list of the contributing brain objects

freq = sys.argv[1]

results_dir = config['resultsdir']

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


data_dir = config['datadir']


def electrode_search(fname, threshold=10):
    # searching original .bo
    fname = os.path.basename(os.path.splitext(fname)[0])
    fname = os.path.join(config['og_bodir'], fname.split('_' + freq)[0] + '.bo')
    kurt_vals = se.load(fname, field='kurtosis')
    thresh_bool = kurt_vals > threshold
    locs = se.load(fname, field='locs')
    if sum(~thresh_bool) > 1:
        locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])
        return locs[~thresh_bool]

union_locs = []
model_data = []

brain_data = glob.glob(os.path.join(data_dir, '*' + freq + '.bo'))

if freq in 'raw':
    brain_data = glob.glob(os.path.join(config['og_bodir'], '*.bo'))

bos_used = []

for i in brain_data:
    try:
        locs = electrode_search(i)
        if not locs.empty:
            if union_locs == []:
                print(os.path.basename(i))
                union_locs = locs.values
                model_data.append(os.path.basename(i))
            else:
                union_locs = np.vstack((union_locs, locs.values))
                model_data.append(os.path.basename(i))
            bos_used.append(i)
    except:
        pass

print(freq)
print(len(bos_used))
print(str([os.path.basename(i) for i in set(brain_data)-set(bos_used)]))

locations, l_indices = _unique(union_locs)

results = os.path.join(results_dir, freq + '_locs.npz')

np.savez(results, locs = locations, subjs = model_data)