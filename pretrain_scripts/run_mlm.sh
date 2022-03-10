#!/bin/bash

# source ~/anaconda3/bin/activate source_transformers

DEVICE=$1

if [ $DEVICE = "0" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    export CUDA_VISIBLE_DEVICES=$1
fi

echo $CUDA_VISIBLE_DEVICES

MODEL_NAME=$2

MIX=$3

BATCH_SIZE=$4

LR=5e-5

EPOCH=$5

DUPE=$6

DATA_PATH=pretrain_data/$DUPE*replace_$MIX.txt
MODEL_PATH=$MODEL_NAME

OUT_PATH=checkpoints/$MODEL_NAME-b$BATCH_SIZE-l$LR-e$EPOCH-m$MIX-d$DUPE
echo $OUT_PATH

python pretrain_scripts/run_mlm.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $DATA_PATH \
    --do_train \
    --output_dir $OUT_PATH \
    --num_train_epochs $EPOCH \
    --overwrite_output_dir \
    --per_device_train_batch_size $BATCH_SIZE\
    --save_total_limit 3 \
    --warmup_ratio 0.06 \
    --learning_rate $LR \
    --fp16 True \
    --line_by_line True\
    --save_strategy steps\
    --save_steps 5000\
    --logging_dir $OUT_PATH/logging\


# source ~/anaconda3/bin/activate transformers
python evaluate_model.py --model-path $OUT_PATH --model-name $MODEL_NAME --task data_evaluation