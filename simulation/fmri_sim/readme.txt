FMRI analysis:

1) Run compile_bo_locs.py

This script compiles 3 things.
 - A list of locations from brain objects in pyfr analysis
 - The union of those locations
 - The gray matter masked nifti image downsampled to 3mm

 To run this, submit `qsub compile_bo_locs.pbs`

 2) Run convert_fmri_bo.py

 This script runs the nii2cmu for the fmri data.

 3) Run fmri_subsample.py

 This script subsamples the fmri data to the pyfr patient locations.

 4) Run compile_mo_locs.py

 Compiles the model locations.

 5) Run fmri_sim_models.py

 Expand to model locations

 6) Run fmri_sim_ave_model.py

 Average the models

 7)