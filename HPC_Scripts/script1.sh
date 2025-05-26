#!/bin/bash
#BSUB -J Graph_ML_Bachelor
#BSUB -o HPC_outputs/ML_bachelor_%J.out
#BSUB -e HPC_outputs/ML_bachelor_%J.err
#BSUB -q gpuv100
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -u s224197@dtu.dk

export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/Packages:$(pwd)/dataset 

nvidia-smi 

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

python src/model1/embed_batches_2.py --alpha=0.1 --epochs=50 --batch_size=64 --embedding_dim=8 --weight=0.10 --venue_weight=100 > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
# python src/model1/embed_valid.py > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
# python src/model1/predict.py > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
 