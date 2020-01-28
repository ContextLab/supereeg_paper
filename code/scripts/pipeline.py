import supereeg as se
import numpy as np
from time import sleep
from glob import glob
from os.path import exists, join
import sh

"""
WIP, automated pipeline for whole analysis
"""

### IMPORTANT CONFIG
og_bodir = '/dartfs/rc/lab/D/DBIC/CDL/f002s72/RAM_analysis/bos'
num_models = 79


def check(update, check, step, wait_time=60, timeout=60*20):
    newest_file = 0
    n = update()
    while not check():
        if newest_file > timeout:
            print('something broken with ' + step)
            exit()
        if n < update():
            newest_file = 0
        else:
            newest_file += wait_time
        n = update()
        sleep(wait_time)

run = sh.Command('python')

freqdir = '/dartfs/rc/lab/D/DBIC/CDL/f003f64/freqs'
num_orig = len(glob(join(og_bodir, '*.bo')))
num_freq = len(glob(join(freqdir, '*.bo')))

if num_freq < 6 * num_orig:
    run('/dartfs-hpc/rc/home/4/f003f64/supereeg_paper/code/scripts/make_freqs/make_freq_bos_submit.py')
    check(lambda: len(glob(join(freqdir, '*.bo'))), \
        lambda: len(glob(join(freqdir, '*.bo'))) == 6 * num_orig, 'freq bo creation')

print('freq creation done')

locs_dir = '/dartfs/rc/lab/D/DBIC/CDL/f003f64/results'
num_locs = len(glob(join(locs_dir, '*locs.npz')))
if num_locs < 6:
    run('/dartfs-hpc/rc/home/4/f003f64/supereeg_paper/code/scripts/pyFR_locs/union_locs_job_submit.py')
    check(lambda: len(glob(join(locs_dir, '*locs.npz'))),\
        lambda: len(glob(join(locs_dir, '*locs.npz'))) == 6, 'locs')

print('locs done')

loc_fs = glob(join(locs_dir, '*locs.npz'))
arr = np.load(loc_fs[0])['locs']
for f in loc_fs:
    arr2 = np.load(f)['locs']
    if not np.array_equal(arr, arr2):
        print('something wrong with locs')
        exit()

results_dir = '/dartfs/rc/lab/D/DBIC/CDL/f003f64/'

locs_dir = join(results_dir, 'union')
def check_freqs():
    freqs = ['delta', 'theta', 'alpha', 'beta', 'lgamma', 'hgamma']
    full_mats = [glob(join(locs_dir, '*'+freq+'*')) for freq in freqs]
    for freq_mats in full_mats:
        if len(freq_mats) != num_models:
            return False
    return True

if not check_freqs():
    run('/dartfs-hpc/rc/home/4/f003f64/supereeg_paper/code/scripts/full_mats/full_mats_job_submit.py')
    check(lambda: len(glob(join(locs_dir, '*'))), lambda: check_freqs, 'full mats')

print('full mats done')

num_ave = lambda: len(glob(join(results_dir,'*.mo')))
if num_ave() < 6:
    run('/dartfs-hpc/rc/home/4/f003f64/supereeg_paper/code/scripts/ave_mats/ave_mats_job_submit.py')
    check(num_ave, lambda: num_ave() < 6, 'ave mats')

print('ave mats done')