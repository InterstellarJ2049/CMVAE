#!/bin/bash

OUTPUTDIR='../outputs'
EXPERIMENT="CUBICC_1_test"
DATADIR='/data/backed_up/shared/Data/CUB/CUBICC'  # ../data
EPOCHS=300
SEED=2
SHARED_LAT_DIM=64
MS_LAT_DIM=32

gpuid=1

# Train CMVAE
CUDA_VISIBLE_DEVICES=${gpuid} python train_CMVAE_CUBICC.py --experiment $EXPERIMENT --obj "dreg" --K 10 --batch-size 32 --epochs $EPOCHS \
      --latent-dim-c 35 --latent-dim-z $SHARED_LAT_DIM --latent-dim-w $MS_LAT_DIM --seed $SEED --beta 1.0 \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --priorposterior 'Normal'
