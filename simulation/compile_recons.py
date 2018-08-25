

from config import config
import os
import glob as glob
import pandas as pd
import numpy as np
import supereeg as se
from supereeg.helpers import _z2r, _r2z


recon_dir = os.path.join(config['recon_datadir'])


results_dir = os.path.join(config['compiled_datadir'])

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

recon_data = glob.glob(os.path.join(recon_dir , '*.npz'))
all_corrs = pd.DataFrame()

for i in recon_data:
    corr_data = np.load(i, mmap_mode='r')
    tempsub = os.path.basename(i)[16:-4]
    tempcorr = _r2z(corr_data['corrs'])
    tempR = corr_data['locs']
    temp_pd = pd.DataFrame()
    for l, item in enumerate(tempR):
        temp_pd = temp_pd.append(pd.DataFrame({'R': [item], 'Correlation': [tempcorr[l]], 'Subject': [tempsub]}))

    if all_corrs.empty:
        all_corrs = temp_pd
    else:
        all_corrs = all_corrs.append(temp_pd)

all_corrs = os.path.join(results_dir, 'all_corrs_fmri_sim.csv')
all_corrs.to_csv(all_corrs)