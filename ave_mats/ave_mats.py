import supereeg as se
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from config import config


model_template = sys.argv[1]

vox_size = sys.argv[2]

model_dir = os.path.join(config['datadir'], model_template)

results_dir = os.path.join(config['resultsdir'], model_template)

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

model_data = glob.glob(os.path.join(model_dir,'*.mo'))


ave_model = se.model_compile(model_data)

ave_model.save(fname=os.path.join(results_dir, model_template))

ave_model.plot_data()

plt.savefig(os.path.join(fig_dir, model_template))


print(ave_model.n_subs)