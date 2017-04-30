#!/bin/bash
#PBS -l walltime=120:00:00
#PBS -l nodes=ngpu-a4-09:ppn=2 -l mem=12gb
#PBS -o /home2/ift6ed13/results/log.out
#PBS -r n

module add python/3.5.1
module add openblas/0.2.18
module add CUDA/7.5

cd /home2/ift6ed13/

cp -rf /home2/ift6ed13/data $LSCRATCH
cp /home2/ift6ed13/code/Launch.py $LSCRATCH
cp /home2/ift6ed13/code/LSGAN.py $LSCRATCH

cd $LSCRATCH

source /home2/ift6ed13/p3.5/bin/activate

python Launch.py

rm -rf *

