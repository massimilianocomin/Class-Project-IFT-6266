#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=2 -l mem=12gb
#PBS -r n

module add python/3.5.1
module add openblas/0.2.18
module add CUDA/7.5

source /home2/ift6ed13/p3.5/bin/activate

python /home2/ift6ed13/code/Launch.py
