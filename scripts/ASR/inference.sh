
DATASET=/name/of/dataset
DATA_ROOT=/path/to/dataset
LABEL_DIR=/path/to/labels
CHECKPOINT_PATH=/path/to/finetuned/model

SUBSET=dev|test
BPE_TOKENIZER=/path/to/tokenizer

USER_DIR=/path/to/artst
RESULTS_PATH=

mkdir -p ${RESULTS_PATH}
 
BEAM=5
CTC_WEIGHT=0.25
MAX_TOKENS=350000

fairseq-generate ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --user-dir ${USER_DIR} \
  --task artst \
  --t5-task s2t \
  --path ${CHECKPOINT_PATH} \
  --hubert-label-dir ${LABEL_DIR} \
  --ctc-weight ${CTC_WEIGHT} \
  --max-tokens ${MAX_TOKENS} \
  --beam ${BEAM} \
  --scoring wer \
  --max-len-a 0 \
  --max-len-b 1000 \
  --sample-rate 16000 \
  --batch-size 1 \
  --num-workers 4 \
  --results-path ${RESULTS_PATH} 
  

grep "^D\-" ${RESULTS_PATH}/generate-${SUBSET}.txt | \
sed 's/^D-//ig' | sort -nk1 | cut -f3 \
> ${RESULTS_PATH}/${SUBSET}-pred.txt

grep "^T\-" ${RESULTS_PATH}/generate-${SUBSET}.txt | \
sed 's/^T-//ig' | sort -nk1 | cut -f2 \
> ${RESULTS_PATH}/${SUBSET}-true.txt

