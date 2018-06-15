import supereeg as se
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from config import config



model_template = sys.argv[1]

radius = sys.argv[2]

model_dir = os.path.join(config['datadir'],  model_template +"_"+ radius)

results_dir = os.path.join(config['resultsdir'],  model_template +"_"+ radius)

locs_file = os.path.join(config['pyFRlocsdir'], 'locs.npz')
R = np.load(locs_file)['locs']

def z2r(z):

    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def _to_log_complex(X):
    """
    Compute the log of the given numpy array.  Store all positive members of the original array in the real component of
    the result and all negative members of the original array in the complex component of the result.
    Parameters
    ----------
    X : numpy array to take the log of
    Returns
    ----------
    log_X_complex : The log of X, stored as complex numbers to keep track of the positive and negative parts
    """
    signX = np.sign(X)
    posX = np.log(np.multiply(signX > 0, X))
    negX = np.log(np.abs(np.multiply(signX < 0, X)))

    negX = np.multiply(0+1j, negX)
    negX.real[np.isnan(negX)] = 0

    return posX + negX

def _to_exp_real(C):
    """
    Inverse of _to_log_complex
    """
    posX = C.real
    negX = C.imag
    return np.exp(posX) - np.exp(negX)

def _set_numerator(n_real, n_imag):
    """
    Internal function for setting the numerator (deals with size mismatches)
    """
    numerator = np.zeros_like(n_real, dtype=np.complex128)
    numerator.real = n_real
    numerator.imag = n_imag

def _recover_model(num, denom, z_transform=False):

    m = np.divide(_to_exp_real(num), np.exp(denom)) #numerator and denominator are in log units
    if z_transform:
        np.fill_diagonal(m, np.inf)
        return m
    else:
        np.fill_diagonal(m, 1)
        return z2r(m)

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
   print(err)


files =glob.glob(os.path.join(model_dir, '*.npz'))

model_data = []

results_1 = []
results_n = []
results_d = []
results_l_n = []
results_l_d = []

count = 0

for i in files:
    count += 1
    data = np.load(i, mmap_mode='r')
    ## need to do this with complex numbers

    num = data['num']
    den = data['den']
    C_est = np.divide(num, den)
    C_est[np.where(np.isnan(C_est))] = 0
    if np.shape(results_1)[0] == 0:
        mo = se.Model(numerator=num, denominator=den, locs=R, n_subs=1)
        results_1 = C_est
        results_n = num
        results_d = den
        results_l_n = _to_log_complex(num)
        results_l_d = np.log(den)
        model_data.append(os.path.basename(i))
    else:
        mo_0 = se.Model(numerator=num, denominator=den, locs=R, n_subs=1)
        mo.update(mo_0)
        results_1 = results_1 + C_est
        results_n = results_n + num
        results_d = results_d + den
        results_l_n_0 = _to_log_complex(num)
        results_l_n.real = np.logaddexp(results_l_n.real, results_l_n_0.real)
        results_l_n.imag = np.logaddexp(results_l_n.imag, results_l_n_0.imag)
        results_l_d = np.logaddexp(results_l_d,np.log(den))
        model_data.append(os.path.basename(i))


average_matrix_1 = z2r(results_1 /count) + np.eye(np.shape(results_1)[0])

results_2 =_recover_model(_to_log_complex(results_n),  np.log(results_d), z_transform=False)
results_2[np.isnan(results_2)] = 0
average_matrix_2 = results_2

results_3 = np.divide(results_n, results_d)
results_3[np.where(np.isnan(results_3))] = 0
average_matrix_3 = z2r(results_3) + np.eye(np.shape(results_3)[0])

results_4 = _recover_model(results_l_n, results_l_d, z_transform=False)
results_4[np.isnan(results_4)] = 0
average_matrix_4 = results_4

results_5 = mo.get_model()
results_5[np.isnan(results_5)] = 0
average_matrix_5 = results_5

outfile_1 = os.path.join(results_dir, 'ave_mat_1.npz')
np.savez(outfile_1, average_matrix=average_matrix_1, n=count, subjs = model_data)

outfile_2 = os.path.join(results_dir, 'ave_mat_2.npz')
#np.savez(outfile_2, num=results_l_n, den=results_l_d, n=count, subjs = model_data)
np.savez(outfile_2, average_matrix=average_matrix_2, n=count, subjs = model_data)

outfile_3 = os.path.join(results_dir, 'ave_mat_3.npz')
#np.savez(outfile_3, num=results_n, den=results_d, n=count, subjs = model_data)
np.savez(outfile_3, average_matrix=average_matrix_3, n=count, subjs = model_data)

outfile_4 = os.path.join(results_dir, 'ave_mat_4.npz')
#np.savez(outfile_4, num=results_n, den=results_d, n=count, subjs = model_data)
np.savez(outfile_4, average_matrix=average_matrix_4, n=count, subjs = model_data)

outfile_5 = os.path.join(results_dir, 'ave_mat_5.npz')
#np.savez(outfile_4, num=results_n, den=results_d, n=count, subjs = model_data)
np.savez(outfile_5, average_matrix=average_matrix_5, n=count, subjs = model_data)
mo.save(os.path.join(results_dir, 'ave_mat_5'))