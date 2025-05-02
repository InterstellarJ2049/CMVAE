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
CUDA_VISIBLE_DEVICES=${gpuid} python train_CMVAE_CUBICC.py --experiment $EXPERIMENT --obj "iwae" --K 1 --batch-size 32 --epochs $EPOCHS \
      --latent-dim-c 35 --latent-dim-z $SHARED_LAT_DIM --latent-dim-w $MS_LAT_DIM --seed $SEED --beta 1.0 \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --inception_path "${DATADIR}/pt_inception-2015-12-05-6726825d.pth" \
      --priorposterior 'Normal'
