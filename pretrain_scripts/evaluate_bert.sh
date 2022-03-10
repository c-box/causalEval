#!/bin/bash

# source ~/anaconda3/bin/activate source_transformers

export CUDA_VISIBLE_DEVICES=$1

MODEL_NAME=$2

MIX=$3

BATCH_SIZE=$4

LR=5e-5

EPOCH=$5

DUPE=$6

OUT_PATH=checkpoints/$MODEL_NAME-b$BATCH_SIZE-l$LR-e$EPOCH-m$MIX-d$DUPE
echo $OUT_PATH

python evaluate_model.py --model-path $OUT_PATH --model-name $MODEL_NAME --task data_evaluation  --batch-size $[BATCH_SIZE*2]