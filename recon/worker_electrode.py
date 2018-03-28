#!/usr/bin/env python

import numpy as np
import sys
import supereeg as se



def main(fname):
    if fname.split('.')[-1] == 'bo':
        bo = se.load(fname)

        data = np.load(fname, mmap_mode='r')
        n_electrodes = np.shape(data['R'])[0] - 1
        print n_electrodes
        sys.exit(n_electrodes)

        def electrode_search(fname, threshold=10):
            with open(fname, 'rb') as f:
                bo = pickle.load(f)
                thresh_bool = bo.kurtosis > threshold
                if sum(~thresh_bool) < 2:
                    return 0
                else:
                    return sum(~thresh_bool)

        num_elecs = electrode_search(fname)

if __name__ == "__main__":
        main(sys.argv[1])