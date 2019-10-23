#!/bin/bash
#PBS -l nodes=1:ppn=16:xk
#PBS -N 1b
#PBS -l walltime=00:30:00
#PBS -e /u/training/tra216/scratch/hw5/1b/msa2.err
#PBS -o /u/training/tra216/scratch/hw5/1b/msa2.out
#PBS -m bea
#PBS -M junyiz4@illinois.edu
cd /u/training/tra216/scratch/hw5/1b
. /opt/modules/default/init/bash
module load python/2.0.1
#module load cudatoolkit
aprun -n 1 -N 1 python sa2.py
