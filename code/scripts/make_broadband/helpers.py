import supereeg as se
import numpy as np
from scipy.signal import hilbert
import glob as glob
import scipy.io as io
import os
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter
from scipy.spatial.distance import pdist, squareform


def butter_filt(data, filterfreq, samplerate, filter_buffer=2, order=2, filt_type='bandstop', axis=0):
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

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, data)
    # swapping out to filtfilt
    y = filtfilt(b, a, data, axis=axis)
    return y

def power_breakdown(data, f, fs, order=2, axis=0):
    # currently filters on axis = 0, which is GOOD for NONTRANSPOSED data, eg, bo.data.values
    buttered = butter_bandpass_filter(data, f[0], f[1], fs, axis=axis)
    hilberted = hilbert(buttered)
    powered = np.abs(hilberted) ** 2

    return buttered # powered (return either power or just bandpassed)

def array_split(ary, indices_or_sections, axis=0):
    """
        Gives the indices for splitting an array into multiple sub-arrays
        code taken from numpy.array_split and slightly modified
    """
    import numpy.core.numeric as _nx
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.')
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] +
                         extras * [Neach_section+1] +
                         (Nsections-extras) * [Neach_section])
        div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()
        return div_points