
import supereeg as se
from supereeg.model import _bo2model, _recover_model, _to_exp_real
from supereeg.helpers import _logsubexp
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from config import config
from stats import time_by_file_index_chunked_local, z2r, r2z
from bookkeeping import remove_electrode, known_unknown, alter_avemat_1, alter_avemat_2

## load brain object
bo_fname = sys.argv[1]
bo = se.load(bo_fname)

file_name = os.path.basename(os.path.splitext(bo_fname)[0])

npz_infile = sys.argv[1][:-2] + 'npz'
sub_data = np.load(npz_infile)

elec_ind = int(sys.argv[2])

model_template = sys.argv[3]

radius = sys.argv[4]

mo_fname = os.path.join(config['modeldir'], model_template + '_' + radius, file_name + '.npz')
mo = np.load(mo_fname, mmap_mode='r')
mo_log_fname = os.path.join(config['modeldir'], model_template + '_' + radius+ '_log', file_name + '.npz')
mo_log = np.load(mo_log_fname, mmap_mode='r')
mo_mo_fname = os.path.join(config['modeldir'], model_template + '_' + radius+ '_log', file_name + '.mo')
mo_mo = se.load(mo_mo_fname)

ave_dir = os.path.join(config['avedir'], model_template+ '_' + radius)

results_dir = os.path.join(config['resultsdir'], model_template+ '_' + radius)



try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)




locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')

bo.get_filtered_bo()

R = np.load(locs_file)['locs']

## subject's locations
R_K_subj = bo.get_locs().as_matrix()

## get info for held out electrode
electrode = bo.locs.iloc[elec_ind]

## remove electrode and get indices
R_K_removed, other_inds = remove_electrode(R_K_subj, R_K_subj, elec_ind)

## inds after kurtosis threshold: known_inds = known electrodes; unknown_inds = all the rest; rm_unknown_ind = where the removed electrode is located in unknown subset
known_inds, unknown_inds, electrode_ind = known_unknown(R, R_K_removed, R_K_subj, elec_ind)

### case 1:

ave ='ave_mat_1.npz'
Ave_data = np.load(os.path.join(ave_dir, ave), mmap_mode='r')

across_dir = os.path.join(results_dir, 'across_subjects_1')

try:
    if not os.path.exists(across_dir):
        os.makedirs(across_dir)
except OSError as err:
    print(err)

recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
if not os.path.isfile(recon_outfile_across):
    Model_across, count = alter_avemat_2(Ave_data, mo)

    Model_across[np.where(np.isnan(Model_across))] = 0
    Model = Model_across + np.eye(np.shape(Model_across)[0])

    corrs = time_by_file_index_chunked_local(npz_infile, Model, known_inds, unknown_inds, electrode_ind, other_inds,
                                             elec_ind, time_series=False)
    print(corrs)

    np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
else:
    print('across model completed')

### case 2:

ave ='ave_mat_2.npz'
Ave_data = np.load(os.path.join(ave_dir, ave), mmap_mode='r')

across_dir = os.path.join(results_dir, 'across_subjects_2')

try:
    if not os.path.exists(across_dir):
        os.makedirs(across_dir)
except OSError as err:
    print(err)

recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
if not os.path.isfile(recon_outfile_across):
    Model_across, count = alter_avemat_1(Ave_data, mo)

    Model_across[np.where(np.isnan(Model_across))] = 0
    Model = Model_across + np.eye(np.shape(Model_across)[0])

    corrs = time_by_file_index_chunked_local(npz_infile, Model, known_inds, unknown_inds, electrode_ind, other_inds,
                                             elec_ind, time_series=False)
    print(corrs)

    np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
else:
    print('across model completed')

### case 3:

ave ='ave_mat_3.npz'
Ave_data = np.load(os.path.join(ave_dir, ave), mmap_mode='r')

across_dir = os.path.join(results_dir, 'across_subjects_3')

try:
    if not os.path.exists(across_dir):
        os.makedirs(across_dir)
except OSError as err:
    print(err)

recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
if not os.path.isfile(recon_outfile_across):
    Model_across, count = alter_avemat_1(Ave_data, mo)

    Model_across[np.where(np.isnan(Model_across))] = 0
    Model = Model_across + np.eye(np.shape(Model_across)[0])

    corrs = time_by_file_index_chunked_local(npz_infile, Model, known_inds, unknown_inds, electrode_ind, other_inds,
                                             elec_ind, time_series=False)
    print(corrs)

    np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
else:
    print('across model completed')

### case 4 best working:

ave ='ave_mat_4.npz'
Ave_data = np.load(os.path.join(ave_dir, ave), mmap_mode='r')

across_dir = os.path.join(results_dir, 'across_subjects_4')

try:
    if not os.path.exists(across_dir):
        os.makedirs(across_dir)
except OSError as err:
    print(err)

recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
if not os.path.isfile(recon_outfile_across):
    Model_across, count = alter_avemat_1(Ave_data, mo)

    Model_across[np.where(np.isnan(Model_across))] = 0
    Model = Model_across + np.eye(np.shape(Model_across)[0])

    corrs = time_by_file_index_chunked_local(npz_infile, Model, known_inds, unknown_inds, electrode_ind, other_inds,
                                             elec_ind, time_series=False)
    print(corrs)

    np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
else:
    print('across model completed')


### case 5:

ave ='ave_mat_5.npz'
Ave_data = np.load(os.path.join(ave_dir, ave), mmap_mode='r')

across_dir = os.path.join(results_dir, 'across_subjects_5')

try:
    if not os.path.exists(across_dir):
        os.makedirs(across_dir)
except OSError as err:
    print(err)

recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
if not os.path.isfile(recon_outfile_across):

    Model_across, count = alter_avemat_2(Ave_data, mo_log)

    Model_across[np.where(np.isnan(Model_across))] = 0
    Model = Model_across + np.eye(np.shape(Model_across)[0])

    corrs = time_by_file_index_chunked_local(npz_infile, Model, known_inds, unknown_inds, electrode_ind, other_inds,
                                             elec_ind, time_series=False)
    print(corrs)

    np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
else:
    print('across model completed')


### case 6:

ave ='ave_mat_6.npz'
Ave_data = np.load(os.path.join(ave_dir, ave), mmap_mode='r')

across_dir = os.path.join(results_dir, 'across_subjects_6')

try:
    if not os.path.exists(across_dir):
        os.makedirs(across_dir)
except OSError as err:
    print(err)

recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
if not os.path.isfile(recon_outfile_across):

    Model_across, count = alter_avemat_2(Ave_data, mo_log)

    Model_across[np.where(np.isnan(Model_across))] = 0
    Model = Model_across + np.eye(np.shape(Model_across)[0])

    corrs = time_by_file_index_chunked_local(npz_infile, Model, known_inds, unknown_inds, electrode_ind, other_inds,
                                             elec_ind, time_series=False)
    print(corrs)

    np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
else:
    print('across model completed')


### case 7:

ave ='ave_mat_6.mo'
mo = se.load(os.path.join(ave_dir, ave))

across_dir = os.path.join(results_dir, 'across_subjects_7')

try:
    if not os.path.exists(across_dir):
        os.makedirs(across_dir)
except OSError as err:
    print(err)

recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
if not os.path.isfile(recon_outfile_across):

    mo_mo._set_numerator(mo_mo.numerator.real, mo_mo.numerator.imag)
    num_sub = _logsubexp(mo.numerator, mo_mo.numerator)
    denom_sub = _to_exp_real(_logsubexp(mo.denominator, mo_mo.denominator))
    Model_across =_recover_model(num_sub,  np.log(denom_sub), z_transform=False)
    Model_across[np.where(np.isnan(Model_across))] = 0
    Model = Model_across + np.eye(np.shape(Model_across)[0])

    corrs = time_by_file_index_chunked_local(npz_infile, Model, known_inds, unknown_inds, electrode_ind, other_inds,
                                             elec_ind, time_series=False)
    print(corrs)

    np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
else:
    print('across model completed')


### case 8:

ave ='ave_mat_6.mo'
mo = se.load(os.path.join(ave_dir, ave))

across_dir = os.path.join(results_dir, 'across_subjects_8')

try:
    if not os.path.exists(across_dir):
        os.makedirs(across_dir)
except OSError as err:
    print(err)

recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
if not os.path.isfile(recon_outfile_across):

    mo_mo._set_numerator(mo_mo.numerator.real, mo_mo.numerator.imag)
    num_sub = _logsubexp(mo.numerator, mo_mo.numerator)
    denom_sub = _to_exp_real(_logsubexp(mo.denominator, mo_mo.denominator))
    Model_across =_recover_model(num_sub,  np.log(denom_sub), z_transform=False)
    Model_across[np.where(np.isnan(Model_across))] = 0
    Model = Model_across + np.eye(np.shape(Model_across)[0])

    corrs = time_by_file_index_chunked_local(npz_infile, Model, known_inds, unknown_inds, electrode_ind, other_inds,
                                             elec_ind, time_series=False)
    print(corrs)

    np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
else:
    print('across model completed')
# ## load brain object
# bo_fname = sys.argv[1]
# bo = se.load(bo_fname)
#
# file_name = os.path.basename(os.path.splitext(bo_fname)[0])
#
# npz_infile = sys.argv[1][:-2] + 'npz'
# sub_data = np.load(npz_infile)
#
# elec_ind = int(sys.argv[2])
#
# model_template = sys.argv[3]
#
# radius = sys.argv[4]
#
# mo_fname = os.path.join(config['modeldir'], model_template + '_' + radius, file_name + '.npz')
# mo = np.load(mo_fname, mmap_mode='r')
#
#
# ave_dir = os.path.join(config['avedir'], model_template+ '_' + radius)
#
# results_dir = os.path.join(config['resultsdir'], model_template+ '_' + radius)
#
# across_dir = os.path.join(results_dir, 'across_subjects')
# within_dir = os.path.join(results_dir, 'within_subjects')
# all_dir = os.path.join(results_dir, 'all_subjects')
#
# try:
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
# except OSError as err:
#    print(err)
#
# try:
#     if not os.path.exists(across_dir):
#         os.makedirs(across_dir)
# except OSError as err:
#    print(err)
#
# try:
#     if not os.path.exists(within_dir):
#         os.makedirs(within_dir)
# except OSError as err:
#    print(err)
#
# try:
#     if not os.path.exists(all_dir):
#         os.makedirs(all_dir)
# except OSError as err:
#    print(err)
#
#
# locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')
#
# bo.get_filtered_bo()
#
# R = np.load(locs_file)['locs']
#
# Ave_data = np.load(os.path.join(ave_dir, 'ave_mat.npz'), mmap_mode='r')
#
# ## subject's locations
# R_K_subj = bo.get_locs().as_matrix()
#
# ## get info for held out electrode
# electrode = bo.locs.iloc[elec_ind]
#
# ## remove electrode and get indices
# R_K_removed, other_inds = remove_electrode(R_K_subj, R_K_subj, elec_ind)
#
# ## inds after kurtosis threshold: known_inds = known electrodes; unknown_inds = all the rest; rm_unknown_ind = where the removed electrode is located in unknown subset
# known_inds, unknown_inds, electrode_ind = known_unknown(R, R_K_removed, R_K_subj, elec_ind)
#
# ### across subjects:
#
# recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
# if not os.path.isfile(recon_outfile_across):
#     Model_across, count = alter_avemat(Ave_data, mo)
#
#     Model_across[np.where(np.isnan(Model_across))] = 0
#     Model = Model_across + np.eye(np.shape(Model_across)[0])
#
#     corrs = time_by_file_index_chunked_local(npz_infile, Model_across, known_inds, unknown_inds, electrode_ind, other_inds,
#                                              elec_ind, time_series=False)
#
#     np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
# else:
#     print('across model completed')
#
# ### all subjects:
# recon_outfile_all = os.path.join(all_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
# if not os.path.isfile(recon_outfile_all):
#     Model_all = Ave_data['average_matrix']
#     Model_all[np.where(np.isnan(Model_all))] = 0
#
#     corrs = time_by_file_index_chunked_local(npz_infile, Model_all, known_inds, unknown_inds, electrode_ind, other_inds,
#                                                 elec_ind, time_series=False)
#
#     np.savez(recon_outfile_all, coord=electrode, corrs=corrs)
# else:
#     print('all model completed')
#
### within subjects:
# recon_outfile_within = os.path.join(within_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
# if not os.path.isfile(recon_outfile_within):
#     bo_sliced = bo[:, other_inds]
#     bo_sliced.filter = None
#     num_corrmat_x, denom_corrmat_x, n_subs = _bo2model(bo_sliced, R_K_subj, 20)
#
#     C_est=np.divide(num_corrmat_x, denom_corrmat_x)
#     C_est[np.where(np.isnan(C_est))] = 0
#     sub_model = z2r(C_est) + np.eye(np.shape(C_est)[0])
#
#     known_inds, unknown_inds, electrode_ind = known_unknown(R_K_subj, R_K_removed, R_K_subj, elec_ind)
#
#     corrs = time_by_file_index_chunked_local(npz_infile, sub_model, known_inds, unknown_inds, electrode_ind, other_inds,
#                                                 elec_ind, time_series=False)
#
#     np.savez(recon_outfile_within, coord=electrode, corrs=corrs)
#
# else:
#     print('within model completed')