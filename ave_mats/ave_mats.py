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
outfile = os.path.join(results_dir, 'ave_mat.npz')

results = []
count = 0
for i in files:
    count += 1
    data = np.load(i, mmap_mode='r')
    C_est = data['C_est']
    C_est[np.where(np.isnan(C_est))] = 0
    if np.shape(results)[0] == 0:
        results = C_est
    else:
        results = results + C_est

average_matrix = z2r(results /count) + np.eye(np.shape(results)[0])
np.savez(outfile, average_matrix=average_matrix, n=count)
