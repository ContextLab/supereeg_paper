import os
import socket

config = dict()

config['template'] = 'run_job.sh'

# ====== MODIFY ONLY THE CODE BETWEEN THESE LINES ======
if (socket.gethostname() == 'Lucys-MacBook-Pro-3.local') or (socket.gethostname() == 'vertex.kiewit.dartmouth.edu') or (socket.gethostname() == 'vertex.local'):
    config['bo_datadir'] = '/Users/lucyowen/Desktop/supereeg_env/bo/'
    config['fmri_datadir'] = '/Users/lucyowen/Desktop/supereeg_env/simulations/fmri/'
    config['bof_datadir'] = '/Users/lucyowen/Desktop/supereeg_env/simulations/bo/'
    config['bof_sub_datadir'] = '/Users/lucyowen/Desktop/supereeg_env/simulations/bo_sub/'
    config['model_datadir'] = '/Users/lucyowen/Desktop/supereeg_env/simulations/models/'
    config['ave_datadir'] = '/Users/lucyowen/Desktop/supereeg_env/simulations/ave_mats/'
    config['recon_datadir'] = '/Users/lucyowen/Desktop/supereeg_env/simulations/recons/'
    config['locs_resultsdir'] = '/Users/lucyowen/Desktop/supereeg_env/simulations/fmri_locs/'
    config['datadir'] = '/Users/lucyowen/Desktop/supereeg_env/simulations'
    config['workingdir'] = config['datadir']
    config['startdir'] = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # directory to start the job in
    config['template'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_job_local.sh')
else:
    config['bo_datadir'] = '/idata/cdl/data/ECoG/pyFR/data/bo/'
    config['fmri_datadir'] = '/idata/cdl/data/fMRI/pieman/nii_files'
    config['bof_datadir'] = '/idata/cdl/lowen/simulations/bo/'
    config['bof_sub_datadir'] = '/idata/cdl/lowen/simulations/bo_sub/'
    config['model_datadir'] = '/idata/cdl/lowen/simulations/models/'
    config['ave_datadir'] = '/idata/cdl/lowen/simulations/ave_mats/'
    config['recon_datadir'] = '/idata/cdl/lowen/simulations/recons/'
    config['locs_resultsdir'] = '/idata/cdl/lowen/simulations/fmri_locs/'
    config['datadir'] = '/idata/cdl/lowen/simulations'
    config['workingdir'] = '/idata/cdl/lowen/simulations'
    config['startdir'] = '/idata/cdl/lowen'
    config['template'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_job.sh')

# job creation options
config['scriptdir'] = os.path.join(config['workingdir'], 'scripts')
config['lockdir'] = os.path.join(config['workingdir'], 'locks')
config['resultsdir'] = os.path.join(config['workingdir'], 'results')

# runtime options
config['jobname'] = "simulation"  # default job name
config['q'] = "default"  # options: default, testing, largeq
config['nnodes'] = 1  # how many nodes to use for this one job
config['ppn'] = 1  # how many processors to use for this one job (assume 4GB of RAM per processor)
config['walltime'] = '30:00:00'  # maximum runtime, in h:MM:SS
#config['startdir'] = '/ihome/lowen/repos/supereeg/examples'  # directory to start the job in
config['cmd_wrapper'] = "python"  # replace with actual command wrapper (e.g. matlab, python, etc.)
config['modules'] = "(\"python/2.7.11\")"  # separate each module with a space and enclose in (escaped) double quotes
# ====== MODIFY ONLY THE CODE BETWEEN THESE LINES ======
