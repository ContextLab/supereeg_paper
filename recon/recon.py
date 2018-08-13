
import supereeg as se
from supereeg.helpers import _corr_column, get_rows, known_unknown, remove_electrode
import numpy as np
import sys
import os
from config import config

bo_fname = sys.argv[1]
bo = se.load(bo_fname)

file_name = os.path.basename(os.path.splitext(bo_fname)[0])

elec_ind = int(sys.argv[2])

model_template = sys.argv[3]

radius = sys.argv[4]

mo_mo_fname = os.path.join(config['modeldir'], model_template + '_' + radius, file_name + '.mo')
mo_mo = se.load(mo_mo_fname)

ave_dir = os.path.join(config['avedir'], model_template+ '_' + radius)

ave ='ave_mat.mo'
mo = se.load(os.path.join(ave_dir, ave))

results_dir = os.path.join(config['resultsdir'], model_template+ '_' + radius)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


bo.apply_filter()

electrode = bo.get_locs().iloc[elec_ind]

R = mo.get_locs().as_matrix()

R_K_subj = bo.get_locs().as_matrix()

R_K_removed, other_inds = remove_electrode(R_K_subj, R_K_subj, elec_ind)

known_inds, unknown_inds, e_ind = known_unknown(R, R_K_removed, R_K_subj, elec_ind)

electrode_ind = get_rows(R, electrode.values)

bo_s = bo[:, other_inds]

mo_s = mo - mo_mo

actual = bo[:,elec_ind]

bo_r = mo_s.predict(bo_s, recon_loc_inds=e_ind)

c = _corr_column(bo_r.data.as_matrix(), actual.get_zscore_data())

print(c)

recon_outfile_across = os.path.join(results_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
np.savez(recon_outfile_across, coord=electrode, corrs=c)


### need to change code below for within and across experiments/subjects for RAM data
# # ## load brain object
# # bo_fname = sys.argv[1]
# # bo = se.load(bo_fname)
# #
# # file_name = os.path.basename(os.path.splitext(bo_fname)[0])
# #
# # npz_infile = sys.argv[1][:-2] + 'npz'
# # sub_data = np.load(npz_infile)
# #
# # elec_ind = int(sys.argv[2])
# #
# # model_template = sys.argv[3]
# #
# # radius = sys.argv[4]
# #
# # mo_fname = os.path.join(config['modeldir'], model_template + '_' + radius, file_name + '.npz')
# # mo = np.load(mo_fname, mmap_mode='r')
# #
# #
# # ave_dir = os.path.join(config['avedir'], model_template+ '_' + radius)
# #
# # results_dir = os.path.join(config['resultsdir'], model_template+ '_' + radius)
# #
# # across_dir = os.path.join(results_dir, 'across_subjects')
# # within_dir = os.path.join(results_dir, 'within_subjects')
# # all_dir = os.path.join(results_dir, 'all_subjects')
# #
# # try:
# #     if not os.path.exists(results_dir):
# #         os.makedirs(results_dir)
# # except OSError as err:
# #    print(err)
# #
# # try:
# #     if not os.path.exists(across_dir):
# #         os.makedirs(across_dir)
# # except OSError as err:
# #    print(err)
# #
# # try:
# #     if not os.path.exists(within_dir):
# #         os.makedirs(within_dir)
# # except OSError as err:
# #    print(err)
# #
# # try:
# #     if not os.path.exists(all_dir):
# #         os.makedirs(all_dir)
# # except OSError as err:
# #    print(err)
# #
# #
# # locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')
# #
# # bo.get_filtered_bo()
# #
# # R = np.load(locs_file)['locs']
# #
# # Ave_data = np.load(os.path.join(ave_dir, 'ave_mat.npz'), mmap_mode='r')
# #
# # ## subject's locations
# # R_K_subj = bo.get_locs().as_matrix()
# #
# # ## get info for held out electrode
# # electrode = bo.locs.iloc[elec_ind]
# #
# # ## remove electrode and get indices
# # R_K_removed, other_inds = remove_electrode(R_K_subj, R_K_subj, elec_ind)
# #
# # ## inds after kurtosis threshold: known_inds = known electrodes; unknown_inds = all the rest; rm_unknown_ind = where the removed electrode is located in unknown subset
# # known_inds, unknown_inds, electrode_ind = known_unknown(R, R_K_removed, R_K_subj, elec_ind)
# #
# # ### across subjects:
# #
# # recon_outfile_across = os.path.join(across_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
# # if not os.path.isfile(recon_outfile_across):
# #     Model_across, count = alter_avemat(Ave_data, mo)
# #
# #     Model_across[np.where(np.isnan(Model_across))] = 0
# #     Model = Model_across + np.eye(np.shape(Model_across)[0])
# #
# #     corrs = time_by_file_index_chunked_local(npz_infile, Model_across, known_inds, unknown_inds, electrode_ind, other_inds,
# #                                              elec_ind, time_series=False)
# #
# #     np.savez(recon_outfile_across, coord=electrode, corrs=corrs)
# # else:
# #     print('across model completed')
# #
# # ### all subjects:
# # recon_outfile_all = os.path.join(all_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
# # if not os.path.isfile(recon_outfile_all):
# #     Model_all = Ave_data['average_matrix']
# #     Model_all[np.where(np.isnan(Model_all))] = 0
# #
# #     corrs = time_by_file_index_chunked_local(npz_infile, Model_all, known_inds, unknown_inds, electrode_ind, other_inds,
# #                                                 elec_ind, time_series=False)
# #
# #     np.savez(recon_outfile_all, coord=electrode, corrs=corrs)
# # else:
# #     print('all model completed')
# #
# ### within subjects:
# # recon_outfile_within = os.path.join(within_dir, os.path.basename(sys.argv[1][:-3] + '_' + sys.argv[2] + '.npz'))
# # if not os.path.isfile(recon_outfile_within):
# #     bo_sliced = bo[:, other_inds]
# #     bo_sliced.filter = None
# #     num_corrmat_x, denom_corrmat_x, n_subs = _bo2model(bo_sliced, R_K_subj, 20)
# #
# #     C_est=np.divide(num_corrmat_x, denom_corrmat_x)
# #     C_est[np.where(np.isnan(C_est))] = 0
# #     sub_model = z2r(C_est) + np.eye(np.shape(C_est)[0])
# #
# #     known_inds, unknown_inds, electrode_ind = known_unknown(R_K_subj, R_K_removed, R_K_subj, elec_ind)
# #
# #     corrs = time_by_file_index_chunked_local(npz_infile, sub_model, known_inds, unknown_inds, electrode_ind, other_inds,
# #                                                 elec_ind, time_series=False)
# #
# #     np.savez(recon_outfile_within, coord=electrode, corrs=corrs)
# #
# # else:
# #     print('within model completed')