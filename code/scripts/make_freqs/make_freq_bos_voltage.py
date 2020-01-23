import sys
import os
import glob as glob
import supereeg as se
from helpers import power_breakdown, butter_filt, butter_bandpass_filter
import pandas as pd
import numpy as np
from config import config
import traceback

SAMPLE_RATE=200

freq = sys.argv[1]

filterfreqs = np.array([[2,4], [4,8], [8,12], [12,30], [30, 60], [60,SAMPLE_RATE/2 - 1]])
freqnames = ['delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma']

freq_range = filterfreqs[freqnames.index(freq)]

bodir = config['resultsdir']
bo_files = glob.glob(os.path.join(bodir, '*.bo'))

for s, bo_file in enumerate(bo_files):
    try:
        fname = os.path.splitext(os.path.basename(bo_file))[0]

        bo_f_file = os.path.join(config['resultsdir'], fname + '_' + freq + '.bo')

        if not os.path.exists(bo_f_file):

            bo = se.load(bo_file)
            bo.filter = None

            # f_data = butter_filt(bo.data.values, 60, bo.sample_rate[0])
            f_data = power_breakdown(f_data, freq_range, SAMPLE_RATE)

            bo_f = se.Brain(data=f_data, locs=bo.locs, sample_rate=bo.sample_rate,
                                    sessions=bo.sessions.values, kurtosis=bo.kurtosis)

            bo_f.save(bo_f_file)

            print('saving: ' + bo_f_file)

        else:
            print(bo_f_file + ' already exists')


    except:
        print('issue with ' + bo_file)
        traceback.print_exc()
