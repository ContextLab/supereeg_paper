import supereeg as se
from config import config
import os, glob
import numpy as np

"""
Joins split brain objects back into the full brain object
"""

files = glob.glob(os.path.join(config['datadir'], '*.bo'))
files = [os.path.split(x)[1].split('.bo')[0] for x in files]


for fname in files:
    chunks = sorted(glob.glob(os.path.join(config['splitdir'], fname + '_chunk*_broadband.bo')), \
        key=lambda path: int(path.split('_chunk')[1].split('_broad')[0]))
    if len(chunks) == 30:
        datalist = []
        for chunk in chunks:
            bo = se.load(chunk)
            locs = bo.locs
            sample_rate = bo.sample_rate
            datalist.append(bo.data.values)
            del bo
            
        data = np.concatenate(datalist)
        print(data.shape)
        se.Brain(data=data, sample_rate=sample_rate, locs=locs).save(os.path.join(config['resultsdir'], fname + '_broadband.bo'))
        del datalist
