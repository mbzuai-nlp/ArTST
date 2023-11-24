DATASET=/name/of/dataset
DATA_ROOT=ASR/_manifest/$DATASET
LABEL_DIR=ASR/_labels/$DATASET
SAVE_DIR=ASR/_models/$DATASET

TRAIN_SET=train
VALID_SET=valid
BPE_TOKENIZER=/path/to/tokenizer
USER_DIR=/path/to/artst
CHECKPOINT_PATH=/path/to/checkpoint


mkdir -p ${SAVE_DIR}

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 1 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  --fp16 \
  \
  --task artst \
  --t5-task s2t \
  --sample-rate 16000 \
  --num-workers 0 \
  --max-tokens 800000 \
  --update-freq 16 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  \
  --criterion artst \
  --report-accuracy \
  --zero-infinity \
  --ce-weight 0.25 \
  --ctc-weight 0.75 \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-08 \
  --weight-decay 0.01 \
  --lr 0.00006 \
  --lr-scheduler tri_stage \
  --phase-ratio "[0.1, 0.4, 0.5]" \
  --final-lr-scale 0.05 \
  \
  --max-update 2048000 \
  --max-text-positions 600 \
  --required-batch-size-multiple 1 \
  --save-interval-updates 2000 \
  --skip-invalid-size-inputs-valid-test \
  \
  --arch artst_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 104000 \
  \
  --keep-last-epochs 3 \
  --feature-grad-mult 1.0 \
  --best-checkpoint-metric s2t_accuracy \
  --maximize-best-checkpoint-metric \
  --finetune-from-model ${CHECKPOINT_PATH}