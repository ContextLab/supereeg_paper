
import supereeg as se
import os
import glob as glob
from config import config

results_dir = config['bof_datadir']

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

fmri_dir = config['fmri_datadir']

for i in list(range(1, len(os.listdir(config['fmri_datadir']))+1)):


    bo_file = os.path.join(results_dir, 'sub-%d-task-intact1' % i)

    if not os.path.exists(bo_file):

        try:
            ## need to do this for intact1 and intact 2!
            bo = se.Brain(se.load(os.path.join(fmri_dir,'sub-%d' % i, 'func', 'sub-%d-task-intact1' % i + '.nii')))
            bo.save(bo_file)
        except:
            print(bo_file + '_issue')


print('done converting brain objects')