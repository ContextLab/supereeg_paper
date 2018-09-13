

from config import config
import os
import glob as glob
import pandas as pd
import numpy as np
import supereeg as se
from supereeg.helpers import _z2r, _r2z

compare_dir = os.path.join(config['compare_datadir'])


results_dir = os.path.join(config['compiled_datadir'])

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

data = glob.glob(os.path.join(compare_dir , '*.npz'))
all_compare = pd.DataFrame()

for i in data:
    compare_data = np.load(i, mmap_mode='r')
    tempsub = os.path.basename(i)[16:-4]
    temp_d_mo = compare_data['d_mo']
    temp_d_ave = compare_data['d_ave']
    temp_pd = pd.DataFrame()
    for l, item in enumerate(temp_d_mo):

        temp_pd = temp_pd.append(pd.DataFrame({'d_mo': [temp_d_mo[l]], 'd_ave': [temp_d_ave[l]], 'Subject': [tempsub]}))

    if all_compare.empty:
        all_compare = temp_pd
    else:
        all_compare = all_compare.append(temp_pd)

all_compare_fmri_sim = os.path.join(results_dir, 'all_compare_fmri_sim.csv')
all_compare.to_csv(all_compare_fmri_sim)
