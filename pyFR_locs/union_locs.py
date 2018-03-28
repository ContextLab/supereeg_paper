import supereeg as se
import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from supereeg.helpers import sort_unique_locs
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

bo_files = glob.glob(os.path.join(data_dir,'*.bo'))

model_data = []
union_locs = pd.DataFrame()

## compile filtered locations of brain objects that have 2 or more electrodes that pass kurtosis threshold
for b in bo_files:
    values = se.filter_subj(b, return_locs=True)
    if values is None:
        pass
    else:
        meta, locs = values
        union_locs = union_locs.append(locs)
        model_data.append(meta)

locations = sort_unique_locs(union_locs)

filepath=os.path.join(results_dir, 'pyFR_k10_locs.npz')

np.savez(filepath, locs = locations, subjs = model_data)

pdfpath=os.path.join(results_dir, 'pyFR_k10_locs.pdf')

ni_plt.plot_connectome(np.eye(locations.shape[0]), locations, display_mode='lyrz', output_file=pdfpath, node_kwargs={'alpha':0.5, 'edgecolors':None}, node_size=10, node_color = np.ones(locations.shape[0]))

print('done')
