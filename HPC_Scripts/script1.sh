#!/bin/bash
#BSUB -J Graph_ML_Bachelor
#BSUB -o ML_bachelor_%J.out
#BSUB -e ML_bachelor_%J.err
#BSUB -q hpc
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=4]"
#BSUB -W 12:00
#BSUB -n 4
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com


src/embed_batches.py
src/embed_valid_sample.py
src/predict.py