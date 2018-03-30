
import supereeg as se
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import pickle
from config import config


fname = sys.argv[1]

model_template = sys.argv[2]

vox_size = sys.argv[3]

elecs = sys.argv[4]

model_dir = os.path.join(config['datadir'], model_template + vox_size)

results_dir = os.path.join(config['resultsdir'], model_template + vox_size)

fig_dir = os.path.join(results_dir, 'figs')

try:
    os.stat(results_dir)
except:
    os.makedirs(results_dir)

try:
    os.stat(fig_dir)
except:
    os.makedirs(fig_dir)


# load locations for model
gray = se.load(intern(model_template))
gray_locs = gray.locs

file_name = os.path.basename(os.path.splitext(fname)[0])

if fname.split('.')[-1]=='bo':
    bo = se.load(fname)
    mo = os.path.join(results_dir, file_name)

    if se.filter_subj(bo):
        model = se.Model(bo, locs=gray_locs)
        model.save(filepath=os.path.join(results_dir, file_name))
        model.plot()
        plt.savefig(os.path.join(fig_dir, file_name))
        print('done')

    else:
        print(file_name + '_filtered')
else:
    print('unknown file type')

### needs to:
# 1 call up brain object
# 2 call up associated model object
# 3 call up average correlation matrix
# 3.5 zscore brain object
# 4 remove 1 electrode from brain object, store as separate brain object
# 5 remove model object from average correlation matrix
# 6 predict everywhere
# 7 index r_bo at electrode location
# 8 save correlation value




# work around if not brain objects, but :
# if fname.split('.')[-1]=='bo':
#     bo = se.filter_elecs(se.load(fname))
#
# else:
#     print('unknown file type')
#
# if bo.locs.shape[0] > 1:
#
#     model = se.Model(bo, locs = gray_locs)
#
#
#     print('done')
#
# else:
#     print(file_name + 'filtered')

# model_data = []
# bo_files = glob.glob(os.path.join('/Users/lucyowen/Desktop/analysis/bo','*.bo'))
# # bo_files = glob.glob(os.path.join('/idata/cdl/data/ECoG/pyFR/data/bo','*.bo'))
# # for i, b in enumerate(bo_files):
# #     if i < 2:
# #         bo = se.load(b)
# #         model_data.append(se.Brain(data=bo.data, locs=bo.locs))
# #     elif i == 2:
# #         bo = se.load(b)
# #         model_data.append(se.Brain(data=bo.data, locs=bo.locs))
# #         model = se.Model(data=model_data)
# #     else:
# #         bo = se.load(b)
# #         model = model.update(bo)
#
# model = se.Model([se.load(b) for b in bo_files[:2]])
# for b in bo_files[2:]:
#     model = model.update(se.load(b))
#
# #model = se.Model(data=model_data)
#
# print(model.n_subs)
#
# model.save(filepath=os.path.join('/Users/lucyowen/Desktop/analysis/ave_model/pyFR_20mm'))
# # model.save(filepath=os.path.join('/dartfs-hpc/scratch/lowen/ave_model/pyFR_20mm'))