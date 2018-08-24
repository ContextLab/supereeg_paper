FMRI analysis:

1) Run compile_bo_locs.py

This script compiles 3 things.
 - A list of locations from brain objects in pyfr analysis
 - The union of those locations
 - The gray matter masked nifti image downsampled to 3mm

 To run this, submit `qsub compile_bo_locs.pbs`

 2) Run convert_fmri_bo.py

 This script