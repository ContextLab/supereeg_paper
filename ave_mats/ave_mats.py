import supereeg as se
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from config import config


model_template = sys.argv[1]

radius = sys.argv[2]

model_dir = os.path.join(config['datadir'],  model_template +"_"+ radius)

results_dir = os.path.join(config['resultsdir'],  model_template +"_"+ radius)


def z2r(z):

    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


files =glob.glob(os.path.join(model_dir, '*.npz'))

model_data = []

results_1 = []
results_n = []
results_d = []
results_l_n = []
results_l_d = []

count = 0

for i in files:
    count += 1
    data = np.load(i, mmap_mode='r')
    num = data['num']
    den = data['den']
    C_est = np.divide(num, den)
    C_est[np.where(np.isnan(C_est))] = 0
    log_num = np.log(num)
    log_den = np.log(den)
    if np.shape(results_1)[0] == 0:
        results_1 = C_est
        results_n = num
        results_d = den
        results_l_n = log_num
        results_l_d = log_den
        model_data.append(os.path.basename(i))
    else:
        results_1 = results_1 + C_est
        results_n = results_n + num
        results_d = results_d + den
        results_l_n = np.logaddexp(results_l_n,log_num)
        results_l_d = np.logaddexp(results_l_d,log_den)
        model_data.append(os.path.basename(i))

average_matrix_1 = z2r(results_1 /count) + np.eye(np.shape(results_1)[0])

results_2 = np.exp(np.log(results_n) - np.log(results_d))
results_2[np.where(np.isnan(results_2))] = 0
average_matrix_2 = z2r(results_2) + np.eye(np.shape(results_2)[0])

results_3 = np.divide(results_n, results_d)
average_matrix_3 = z2r(results_3) + np.eye(np.shape(results_3)[0])

results_4 = np.exp(results_l_n - results_l_d)
results_4[np.where(np.isnan(results_4))] = 0
average_matrix_4 = z2r(results_4) + np.eye(np.shape(results_4)[0])

outfile_1 = os.path.join(results_dir, 'ave_mat_1.npz')
np.savez(outfile_1, average_matrix=average_matrix_1, n=count, subjs = model_data)

outfile_2 = os.path.join(results_dir, 'ave_mat_2.npz')
np.savez(outfile_2, average_matrix=average_matrix_2, n=count, subjs = model_data)

outfile_3 = os.path.join(results_dir, 'ave_mat_3.npz')
np.savez(outfile_3, average_matrix=average_matrix_3, n=count, subjs = model_data)

outfile_4 = os.path.join(results_dir, 'ave_mat_4.npz')
np.savez(outfile_4, average_matrix=average_matrix_4, n=count, subjs = model_data)
