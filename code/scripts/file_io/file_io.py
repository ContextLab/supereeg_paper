
import supereeg as se
import numpy as np
from supereeg.helpers import tal2mni
import glob
import sys
import os
from config import config

try:
    os.stat(config['resultsdir'])
except:
    os.makedirs(config['resultsdir'])

def npz2bo(infile):

    with open(infile, 'rb') as handle:
        f = np.load(handle)
        f_name = os.path.splitext(os.path.basename(infile))[0]
        data = f['Y']
        sample_rate = f['samplerate']
        sessions = f['fname_labels']
        locs = tal2mni(f['R'])
        meta = f_name

    return se.Brain(data=data, locs=locs, sessions=sessions, sample_rate=sample_rate, meta=meta)

results_dir = config['resultsdir']

RESAMPLERATE=250
fname = sys.argv[1]

file_name = os.path.basename(os.path.splitext(fname)[0])
bo = npz2bo(fname)
bo.resample(resample_rate=RESAMPLERATE)


bo.save(fname=os.path.join(results_dir, file_name))


print('done')
