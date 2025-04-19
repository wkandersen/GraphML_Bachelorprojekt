#!/bin/bash
#BSUB -J Graph_ML_Bachelor
#BSUB -o HPC_outputs/ML_bachelor_%J.out
#BSUB -e HPC_outputs/ML_bachelor_%J.err
#BSUB -q gpua40
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=3GB]"
#BSUB -n 8
#BSUB -W 4:00
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com

export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/Packages:$(pwd)/dataset

nvidia-smi 

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

python src/model1/embed_batches.py > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
python src/model1/embed_valid_sample.py > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
python src/model1/predict.py > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
 