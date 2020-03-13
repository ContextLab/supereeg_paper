import sys
import supereeg as se
import numpy as np
import sklearn
import pycwt as wavelet
import sys
import cProfile, pstats
import io
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Queue, Process
from operator import itemgetter
from scipy.signal import butter, filtfilt
import os
from config import config
from helpers import butter_filt, array_split

def helper(i1, i2, powers_by_freq, qpos, xs, midpoint, mhq): 
        """
        Worker function called by child processes

        i1: int, beginning time index for which the worker is going to calculated broadband power
        i2: int, end time index of the worker
        powers_by_freq: ndarray containing data
                axis 0 is all the frequency bands
                axis 1 is each electrode for each frequency band
                axis 2 is each timepoint for each electrode for each frequency band
        qpos: this is the queue position of the worker, used later for ordering the calculated chunks
        midpoint: float, middle frequency in log-space
        mhq: the multiprocessing Queue to which to add the mean height calculations
        """
        powers_by_freq = powers_by_freq[:,:,i1:i2]
        shape = powers_by_freq.shape[1:]
        mh_data = np.zeros(shape=shape)
        HR = sklearn.linear_model.HuberRegressor(max_iter=50)
        errors = 0

        for timepoint in range(shape[1]):
                for electrode in range(shape[0]):
                        ys = powers_by_freq[:, electrode, timepoint]
                        try: # sometimes, but extremely rarely, the fitting fails
                                HR.fit(xs, ys)
                        except:
                                errors += 1
                        try:
                                mean_height = HR.coef_[0] * midpoint + HR.intercept_
                        except:
                                mean_height = 0
                        mh_data[electrode][timepoint] = mean_height

        mhq.put((mh_data, qpos))


if __name__ == "__main__":

        fname = sys.argv[1]
        bo = se.load(fname)
        data = np.asarray(bo.data.T)
        locs = bo.locs
        sample_rate = bo.sample_rate
        del bo
        
        # filter out 60 Hz + harmonics
        data = butter_filt(data, filterfreq=60, samplerate=sample_rate[0], axis=1)

        # reshape data into 1D arrays for faster FFTs
        og_shape = data.shape
        data = data.ravel()

        # get the log spaced frequencies
        numfreqs = 50
        nyq = np.floor_divide(sample_rate[0], 2.0)
        maxfreq = np.min([100, nyq])
        minfreq = 2
        freqs = np.logspace(np.log10(minfreq), np.log10(maxfreq), num=numfreqs)

        # make an empty ndarray to hold the freq * electrode * timepoint data
        powers_by_freq = np.zeros(shape=(len(freqs), og_shape[0], og_shape[1]))

        # convolve!
        for i, freq in enumerate(freqs):
                wav_transform = wavelet.cwt(data, 1/sample_rate[0], freqs = np.full(1,freq), wavelet=wavelet.Morlet(4))
                # get the power and reshape data back into original shape
                wav_transform = (np.abs(wav_transform[0])**2).reshape(og_shape) 
                powers_by_freq[i] = np.log(wav_transform)

        # prep some variables for the robust regression done in parallel
        xs = np.log(freqs).reshape(-1,1)
        midpoint = (np.log(maxfreq) - np.log(minfreq)) / 2

        nworkers = int(config['nnodes'] * config['ppn'] * 0.5)
        
        # get the indices for the chunks
        chunk_indices = array_split(powers_by_freq, nworkers, axis=2)

        mhq = Queue(nworkers)
        mh_list = []

        processes = [Process(target=helper, args=(chunk_indices[i], chunk_indices[i+1], \
                powers_by_freq, i, xs, midpoint, mhq)) for i in range(nworkers)]

        for p in processes:
                p.start()

        for i in range(nworkers):
                mh_list.append(mhq.get())

        for p in processes:
                p.join()

        # sorting the list of results into original order
        mh_list.sort(key=itemgetter(-1))
        # extracting only the ndarrays from the tuples in mh_list
        for i in range(len(mh_list)):
                mh_list[i] = mh_list[i][0]

        mh_data = np.concatenate(mh_list, axis=1)
        mh_bo = se.Brain(data=mh_data.T, locs=locs, sample_rate=sample_rate)
        mh_bo.save(os.path.join(config['resultsdir'], os.path.split(fname)[1].split('.')[0] + '_broadband.bo'))