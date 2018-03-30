
import supereeg as se
from supereeg.helpers import filter_subj
import numpy as np
import glob
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from config import config


fname = sys.argv[1]

model_template = sys.argv[2]

vox_size = sys.argv[3]

results_dir = os.path.join(config['resultsdir'], model_template + vox_size)

fig_dir = os.path.join(results_dir, 'figs')

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

try:
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
except OSError as err:
   print(err)

# load locations for model
if model_template == 'pyFR_union':
    data = np.load(os.path.join(config['pyFRlocsdir'],'pyFR_k10_locs.npz'))
    locs = data['locs']
    gray_locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

elif model_template == 'example_model':
    gray = se.Brain(se.load('gray', vox_size=20))
    gray_locs = gray.locs

else:
    gray = se.Brain(se.load(intern(model_template), vox_size=int(vox_size)))
    gray_locs = gray.locs


file_name = os.path.basename(os.path.splitext(fname)[0])

if fname.split('.')[-1]=='bo':
    values = filter_subj(fname, return_locs=False)
    if values is None:
        pass
    else:
        bo = se.load(fname)
        model = se.Model(bo, locs=gray_locs)
        model.save(fname=os.path.join(results_dir, file_name))
        model.plot_data()
        plt.savefig(os.path.join(fig_dir, file_name))
        print('done')

else:
    print('unknown file type')

