import supereeg as se
from supereeg.helpers import _unique
import os
import pandas as pd
import numpy as np
import glob as glob
from config import config

results_dir = os.path.join(config['ave_datadir'])

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

mos =glob.glob(os.path.join(config['model_datadir'], '*.mo'))

mo = se.Model(mos, n_subs=len(mos))

mo.save(os.path.join(results_dir, 'sub_locs_ave_model'))

#mo.save(os.path.join(results_dir, 'ave_model'))

