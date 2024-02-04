DATASET=/name/of/dataset
DATA_ROOT=TTS/_manifest/$DATASET
LABEL_DIR=TTS/_labels/$DATASET
RESULTS_PATH=/save/folder
SUBSET=test
BPE_TOKENIZER=/path/to/tokenizer
USER_DIR=/speecht5
CHECKPOINT_PATH=/path/to/checkpoint

mkdir -p ${RESULTS_PATH}

python3 generate_speech.py ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --user-dir ${USER_DIR} \
  --task artst \
  --t5-task t2s \
  --path ${CHECKPOINT_PATH} \
  --hubert-label-dir ${LABEL_DIR} \
  --batch-size 1 \
  --results-path ${RESULTS_PATH} \
  --sample-rate 16000 