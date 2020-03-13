import os
import socket

config = dict()

config['template'] = 'run_job.sh'

# ====== MODIFY ONLY THE CODE BETWEEN THESE LINES ======
if (socket.gethostname() == 'Lucys-MacBook-Pro-3.local') or (socket.gethostname() == 'vertex.kiewit.dartmouth.edu') or (socket.gethostname() == 'vertex.local'):
    config['datadir'] = '/Users/lucyowen/Desktop/seizure_data/edfs'
    config['workingdir'] = '/Users/lucyowen/Desktop/seizure_data'
    config['startdir'] = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # directory to start the job in
    config['template'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_job_local.sh')
else:
    # config['datadir'] = '/dartfs/rc/lab/D/DBIC/CDL/data/ECoG/DHMC_seizure/DHMC_seizure/edfs'
    config['datadir'] = '/dartfs/rc/lab/D/DBIC/CDL/f002s72/RAM_analysis/bos'
    # config['datadir'] = '/dartfs/rc/lab/D/DBIC/CDL/f003f64/'
    config['workingdir'] = '/dartfs/rc/lab/D/DBIC/CDL/f003f64'
    config['startdir'] = '/dartfs/rc/lab/D/DBIC/CDL/f003f64'
    config['template'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_job.sh')

# job creation options
config['splitdir'] = os.path.join(config['workingdir'], 'splits')
config['scriptdir'] = os.path.join(config['workingdir'], 'RAM_scripts')
config['lockdir'] = os.path.join(config['workingdir'], 'RAM_locks')
config['resultsdir'] = os.path.join(config['workingdir'], 'freqs')

# jobst
config['finaldir'] = os.path.join(config['workingdir'], 'jobst')
config['resultsdir'] = os.path.join(config['workingdir'], 'jobst')

# runtime options
config['feature'] = 'cellh|cellk|cellm'
config['jobname'] = "robust_job"  # default job name
config['q'] = "default"  # options: default, testing, largeq
config['nnodes'] = 1  # how many nodes to use for this one job
config['ppn'] = 16  # how many processors to use for this one job (assume 4GB of RAM per processor)
config['walltime'] = '120:00:00'  # maximum runtime, in h:MM:SS
config['cmd_wrapper'] = "python3"  # replace with actual command wrapper (e.g. matlab, python, etc.)
config['modules'] = "(\"python/3.5\")"  # separate each module with a space and enclose in (escaped) double quotes

# ====== MODIFY ONLY THE CODE BETWEEN THESE LINES ======
