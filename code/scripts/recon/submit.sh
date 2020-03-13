#!/bin/bash -l

# DO NOT MODIFY THIS FILE!
# MODIFY config.py AND create_and_submit_jobs.py AS NEEDED

# Portable Batch System (PBS) lines begin with "#PBS".  to-be-replaced text is sandwiched between angled brackets

# declare a name for this job
#PBS -N submittingjobs
# specify the queue the job will be added to (if more than 600, use largeq)
#PBS -q default
# specify the number of cores and nodes (estimate 4GB of RAM per core)
#PBS -l nodes=1:ppn=4
# specify how long the job should run (wall time)
#PBS -l walltime=200:00:00
# set the working directory *of this script* to the directory from which the job was submitted

# set the working directory *of the job* to the specified start directory
cd /dartfs-hpc/rc/home/4/f003f64/supereeg_paper/code/scripts/recon

/optnfs/python/el7/2.7-Anaconda/bin/kinit f003f64@KIEWIT.DARTMOUTH.EDU -k -t /dartfs-hpc/rc/home/4/f003f64/f003f64.keytab -c /dartfs-hpc/rc/home/4/f003f64/krbcache; export KRB5CCNAME="/dartfs-hpc/rc/home/4/f003f64/krbcache"

conda activate supereeg_env

python3 recon_job_submit.py
