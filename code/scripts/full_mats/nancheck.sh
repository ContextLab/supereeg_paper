#!/bin/bash -l

# declare a name for this job
#PBS -N nancheck

# specify the queue the job will be added to (if more than 600, use largeq)
#PBS -q default
# specify the number of cores and nodes (estimate 4GB of RAM per core)
#PBS -l nodes=1:ppn=16

# specify how long the job should run (wall time)
#PBS -l walltime=50:00:00

# set the working directory *of this script* to the directory from which the job was submitted

# set the working directory *of the job* to the specified start directory

echo ACTIVATING supereeg VIRTUAL ENVIRONMENT
conda activate supereeg_env

python nancheck.py
