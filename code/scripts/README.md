# supereeg_paper

To run pyFR analysis:

# 1 file_io:
- converts patient data from npz to brain objects and resample data to 250 Hz

# 2 pyFR_locs
compiles kurtosis thresholded electrodes for only patients with >1 electrodes

# 3 full_mats
expands patient covariance matrices to pyFR_locs (or designated model locations)

# 4 ave_mats
combines correlation matrices to one average model

# 5 recon
reconstruct activity for each electrode
