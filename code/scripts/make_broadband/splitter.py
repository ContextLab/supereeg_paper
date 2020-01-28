import supereeg as se
from config import config
import os, glob
import numpy as np

"""
Splits each brain object into smaller brain objects to avoid memory issues
"""

completed = glob.glob(os.path.join(config['resultsdir'], '*.bo'))
completed_trim_set = set([os.path.split(x)[1].split('_broadband.bo')[0] for x in completed])

all_files = glob.glob(os.path.join(config['datadir'],'*.bo'))
all_files_trim_set = set([os.path.split(x)[1].split('.')[0] for x in all_files])

files = list(all_files_trim_set - completed_trim_set)
files = [os.path.join(config['datadir'], x + '.bo') for x in files]


for fname in files:
    bo = se.load(fname)
    sample_rate = bo.sample_rate
    locs = bo.locs
    data_list = np.array_split(bo.data.values, 30)
    del bo
    for i, data in enumerate(data_list):
        bo = se.Brain(data=data, sample_rate=sample_rate, locs=locs)
        fname = os.path.split(fname)[1].split('.')[0]
        # bo.save(os.path.join(config['splitdir'], os.path.split(fname)[1][:-3] + '_chunk' + str(i)+ '.bo'))
        bo.save(os.path.join(config['splitdir'], fname + '_chunk' + str(i) + '.bo'))
        del bo
