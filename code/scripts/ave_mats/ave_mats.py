import supereeg as se
import numpy as np
import glob
import sys
import os
from config import config

freq = sys.argv[1]

model_dir = os.path.join(config['datadir'])

results_dir = config['resultsdir']
model_dir = config['datadir']

if os.path.exists(os.path.join(results_dir, 'ave_mat_' + freq)):
    print('ave mat already exists')
    exit()

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)

mos = glob.glob(os.path.join(model_dir, '*' + freq + '.mo'))

if freq == 'raw':
    freqnames = ['delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma', 'broadband']
    mos = set(glob.glob(os.path.join(model_dir, '*')))
    for fre in freqnames:
        mos -= set(glob.glob(os.path.join(model_dir, '*' + fre + '*')))
    mos = list(mos)

print(len(mos))

mo = se.Model(mos, n_subs=len(mos))

mo.save(os.path.join(results_dir, 'ave_mat_' + freq))