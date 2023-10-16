#!/bin/bash
DATASET=ALL
MODEL=ALL
CHECKPOINT_PATH=~/TTS/v1.2/models/$DATASET/checkpoint_last.pt
DATA_ROOT=~/TTS/v1.2/hubert_labels/
SUBSET=unseen
BPE_TOKENIZER=~/TTS/arabic.model
LABEL_DIR=~/TTS/v1.2/labels
USER_DIR=~/TTS/artst
RESULTS_PATH=~/TTS/v1.2/results/new/

mkdir -p ${RESULTS_PATH}

python3 TTS/generate_speech.py ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --user-dir ${USER_DIR} \
  --task artst \
  --t5-task t2s \
  --path ${CHECKPOINT_PATH} \
  --hubert-label-dir ${LABEL_DIR} \
  --batch-size 1 \
  --results-path ${RESULTS_PATH} \
  --sample-rate 16000 \
  --inference-speech True