import supereeg as se
import numpy as np
import glob
import sys
import os
from config import config



model_template = sys.argv[1]

radius = sys.argv[2]

model_dir = os.path.join(config['datadir'],  model_template +"_"+ radius)

results_dir = os.path.join(config['resultsdir'],  model_template +"_"+ radius)

locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')
R = np.load(locs_file)['locs']

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

mos =glob.glob(os.path.join(model_dir, '*.mo'))

mo = se.Model(mos, n_subs=len(mos))

mo.save(os.path.join(results_dir, 'ave_mat'))