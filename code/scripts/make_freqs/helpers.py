import supereeg as se
import numpy as np
from scipy.signal import hilbert
#from ptsa.extensions import morlet
import glob as glob
import scipy.io as io
import os
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter
from scipy.spatial.distance import pdist, squareform


def butter_filt(data, filterfreq, samplerate, filter_buffer=2, order=2, filt_type='bandstop', axis=0):
    # used for filtering out 60 Hz + harmonics
    # axis = 0 for non-transposed data, axis=1 for transposed

    filterfreqs = np.arange(filterfreq, np.floor_divide(samplerate, 2.0), filterfreq)

    for f in filterfreqs:
        freq_range = [f - filter_buffer, f + filter_buffer]

        data = buttfilt(data,freq_range, sample_rate=samplerate, filt_type=filt_type, order=order, axis=axis)

    return data

def buttfilt(dat,freq_range,sample_rate,filt_type,order,axis=0):
    """Wrapper for a Butterworth filter.
    """

    # make sure dat is an array
    dat = np.asarray(dat)


    # set up the filter
    freq_range = np.asarray(freq_range)

    # Nyquist frequency
    nyq=sample_rate/2.

    # generate the butterworth filter coefficients
    [b,a]=butter(order,freq_range/nyq,btype=filt_type)

    dat = filtfilt(b,a,dat,axis=axis)

    return dat

def butter_coefs(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    b, a = butter_coefs(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y

def power_breakdown(data, f, fs, order=2, axis=0):
    # currently filters on axis = 0, which is GOOD for NONTRANSPOSED data, eg, bo.data.values
    buttered = butter_bandpass_filter(data, f[0], f[1], fs, axis=axis)
    hilberted = hilbert(buttered)
    powered = np.abs(hilberted)
    return powered