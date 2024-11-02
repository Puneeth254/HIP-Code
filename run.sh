#!/bin/bash
#PBS -o logfile.log
#PBS -e errorfile_slash.err
#PBS -l walltime=00:60:00
#PBS -l select=2:ncpus=32
#PBS -q rupesh_gpuq

# module load openmpi316

hipcc convexHull.hip -o convexHull
./convexHull < testcase.txt > output.txt
