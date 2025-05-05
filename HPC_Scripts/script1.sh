#!/bin/bash
#BSUB -J Graph_ML_Bachelor
#BSUB -o HPC_outputs/ML_bachelor_%J.out
#BSUB -e HPC_outputs/ML_bachelor_%J.err
#BSUB -q c02613
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=6GB]"
#BSUB -n 4
#BSUB -W 00:15
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com

export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/Packages:$(pwd)/dataset

nvidia-smi 

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

python src/model1/embed_batches_2.py --alpha=3 --epochs=100 --batch_size=350 > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
# python src/model1/embed_valid_sample.py > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
# python src/model1/predict.py > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
 