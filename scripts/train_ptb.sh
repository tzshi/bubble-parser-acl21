#!/bin/bash

source ~/environments/python3env/bin/activate

EVAL_STEPS=3200
DECAY_EVALS=5
DECAY_TIMES=2
DECAY_RATIO=0.1

BATCH_SIZE=8

LEARNING_RATE=1e-3
BETA1=0.9
BETA2=0.999
EPSILON=1e-8
WEIGHT_DECAY=0
WARMUP=800

CLIP=5.0

GPU=True

CUTOFF=5

WDIMS=100
EDIMS=100
CDIMS=32
PDIMS=0
IDIMS=16
WORD_DROPOUT=0.3

CHAR_HIDDEN=128
CHAR_DROPOUT=0.3

BILSTM_DIMS=800
BILSTM_LAYERS=3
BILSTM_DROPOUT=0.3

PROJ_DIMS=${BILSTM_DIMS}

TAGGER_DIMS=400
TAGGER_DROPOUT=0.3

PARSER_DIMS=400
PARSER_DROPOUT=0.3

BERT=False

EMBEDDING_FILE=./embeddings/glove.6b.100

TRAIN_FILE=./data/ptb-train.ext
DEV_FILE=./data/ptb-dev.ext

LOG_FOLDER=./models/

mkdir -p $LOG_FOLDER

RUN=ptb

SAVE_PREFIX=${LOG_FOLDER}/${RUN}

mkdir -p $SAVE_PREFIX

OMP_NUM_THREADS=3 \
python3 -m bubble.parser \
    - build-vocab $TRAIN_FILE --cutoff ${CUTOFF} \
    - create-parser --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON --clip $CLIP \
        --wdims $WDIMS --cdims $CDIMS --edims $EDIMS --pdims $PDIMS --idims $IDIMS \
        --word-dropout $WORD_DROPOUT \
        --bilstm-dims $BILSTM_DIMS --bilstm-layers $BILSTM_LAYERS --bilstm-dropout $BILSTM_DROPOUT \
        --proj-dims $PROJ_DIMS \
        --char-hidden $CHAR_HIDDEN --char-dropout $CHAR_DROPOUT \
        --parser-dims $PARSER_DIMS --parser-dropout $PARSER_DROPOUT \
        --tagger-dims $TAGGER_DIMS --tagger-dropout $TAGGER_DROPOUT \
        --weight-decay $WEIGHT_DECAY \
        --warmup $WARMUP \
        --bert $BERT \
        --gpu $GPU \
    - load-embeddings $EMBEDDING_FILE \
    - train $TRAIN_FILE --dev $DEV_FILE \
        --eval-steps $EVAL_STEPS --decay-evals $DECAY_EVALS --decay-times $DECAY_TIMES --decay-ratio $DECAY_RATIO \
        --save-prefix $SAVE_PREFIX/ \
    - finish
