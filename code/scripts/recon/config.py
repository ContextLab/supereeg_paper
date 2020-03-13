import os
import socket

config = dict()

config['template'] = 'run_job.sh'

# ====== MODIFY ONLY THE CODE BETWEEN THESE LINES ======
if (socket.gethostname() == 'Lucys-MacBook-Pro-3.local') or (socket.gethostname() == 'vertex.kiewit.dartmouth.edu') or (socket.gethostname() == 'vertex.local'):
    config['datadir'] = '/Users/lucyowen/Desktop/supereeg_env/bo/'
    config['workingdir'] = '/Users/lucyowen/Desktop/supereeg_env/recon/'
    config['startdir'] = '/Users/lucyowen/Desktop/supereeg_env/'  # directory to start the job in
    config['template'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_job_local.sh')
else:
    config['datadir'] = '/dartfs/rc/lab/D/DBIC/CDL/f003f64/freqs'
    config['workingdir'] = '/dartfs/rc/lab/D/DBIC/CDL/f003f64'
    config['startdir'] = '/dartfs/rc/lab/D/DBIC/CDL/f003f64/'
    config['template'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'run_job.sh')

# job creation options
config['scriptdir'] = os.path.join(config['workingdir'], 'scripts')
config['lockdir'] = os.path.join(config['workingdir'], 'locks')
config['resultsdir'] = os.path.join(config['workingdir'], 'results')
config['modeldir'] = os.path.join(config['resultsdir'], 'union')
config['avedir'] = os.path.join(config['startdir'], 'ave_mats/results')
config['og_bodir'] = '/dartfs/rc/lab/D/DBIC/CDL/f002s72/RAM_analysis/bos'
config['og_bodir'] = '/idata/cdl/data/ECoG/pyFR/data/bo'

# runtime options
config['jobname'] = "recon"  # default job name
config['q'] = "default"  # options: default, testing, largeq
config['nnodes'] = 1  # how many nodes to use for this one job
config['feature'] = 'celln'
config['ppn'] = 16  # how many processors to use for this one job (assume 4GB of RAM per processor)
config['walltime'] = '50:00:00'  # maximum runtime, in h:MM:SS
config['cmd_wrapper'] = "python3"  # replace with actual command wrapper (e.g. matlab, python, etc.)
config['modules'] = "(\"python/3.5\")"  # separate each module with a space and enclose in (escaped) double quotes
# ====== MODIFY ONLY THE CODE BETWEEN THESE LINES ======
