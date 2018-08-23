
import supereeg as se
from supereeg.helpers import _unique
import os
import pandas as pd
import numpy as np
import glob as glob
from config import config

results_dir = config['bof_datadir']

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

fmri_dir = config['fmri_datadir']

niis =glob.glob(os.path.join(config['fmri_datadir'], '*.nii'))


#for i in list(range(1, len(niis)+1)):
for i in list(range(1, 2)):

    bo_file = os.path.join(results_dir+'sub-%d-task-intact1' % i)

    if not os.path.exists(bo_file):
        bo = se.Brain(se.load(os.path.join(fmri_dir,'sub-%d-task-intact1.nii' % i)))
        bo.save(os.path.join(results_dir+'sub-%d-task-intact1' % i))


print('done converting brain objects')