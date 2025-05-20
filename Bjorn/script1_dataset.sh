#!/bin/bash
#BSUB -J Graph_ML_Bachelor
#BSUB -o HPC_outputs/ML_bachelor_%J.out
#BSUB -e HPC_outputs/ML_bachelor_%J.err
#BSUB -q c02613
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -W 00:30
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com

export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/Packages:$(pwd)/dataset:$(pwd)/Bjorn

nvidia-smi 

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

python Bjorn/script1_dataset.sh --embedding_dim=2 --a=100