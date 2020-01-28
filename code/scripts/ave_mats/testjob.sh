#!/bin/bash -l

# DO NOT MODIFY THIS FILE!
# MODIFY config.py AND create_and_submit_jobs.py AS NEEDED

# Portable Batch System (PBS) lines begin with "#PBS".  to-be-replaced text is sandwiched between angled brackets

# declare a name for this job
#PBS -N test
# specify the queue the job will be added to (if more than 600, use largeq)
#PBS -q default

# specify the number of cores and nodes (estimate 4GB of RAM per core)
#PBS -l nodes=1:ppn=2

# specify how long the job should run (wall time)
#PBS -l walltime=10:00

# set the working directory *of this script* to the directory from which the job was submitted

# set the working directory *of the job* to the specified start directory
conda init bash