
import supereeg as se
from supereeg.helpers import _corr_column, get_rows
import numpy as np
import sys
import os
from config import config

## load brain object,
bo_fname = sys.argv[1]

bo = se.load(bo_fname)

file_name = os.path.basename(os.path.splitext(bo_fname)[0])

model_template = sys.argv[2]

radius = sys.argv[3]

vox_size = sys.argv[4]

## might want to do this.. expand the brain object to the model locations and add
# mo_mo_fname = os.path.join(config['modeldir'], model_template + '_' + vox_size, file_name + '.mo')
# mo_mo = se.load(mo_mo_fname)

ave_dir = os.path.join(config['avedir'], model_template+ '_' + vox_size)

results_dir = os.path.join(config['resultsdir'], model_template+ '_' + vox_size)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

ave ='ave_mat.mo'
mo = se.load(os.path.join(ave_dir, ave))

bo.apply_filter()

bo_r = mo.predict(bo, nearest_neighbor=True, force_update=True)

bo_r.save(os.path.join(results_dir, file_name + '_' + model_template + '_' + vox_size + '.bo'))
