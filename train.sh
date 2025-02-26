#!/bin/bash

SPLT_GEN=0
MAX_EPOCHS=30
N_SAMPLES=5
RUNDS="StanfordDogs,StanfordCars,Caltech101,OxfordFlowers"

GEN_DATA=$1
ATTEMPT=$2
SEED=$3
DEBUG=$4
N_SAMPLES=$5
MINSTOPACC=$6
RUNDS=$7
ARCH=$8


DIR=/app/clip_forget_final/results/results${ATTEMPT}/seed_${SEED}/
if [ -e "${DIR}results_${ds}.pkl" ]; then
    echo "Oops! The results exist at '${DIR}results_${ds}.pkl' (so skip this job)"
else
    echo "Saving dir ${DIR}"
    python3 train_forget.py \
    --output_dir /app/clip_forget_final/results/results${ATTEMPT}/seed_${SEED}/ \
    --seed ${SEED} \
    --generated_data ${GEN_DATA} \
    --debug ${DEBUG} \
    --max_epochs ${MAX_EPOCHS} \
    --n_samples ${N_SAMPLES} \
    --run_ds ${RUNDS} \
    --min_train_acc_stop ${MINSTOPACC} \
    --backbone_arch ${ARCH} 
fi
