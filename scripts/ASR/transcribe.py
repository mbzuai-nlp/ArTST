#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Transcribe pre-processed data with a fine-tuned checkpoint.

How 
"""

import ast
import logging
import argparse
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter

import os
import torch
import gradio as gr
import numpy as np
import os.path as op
import pyarabic.araby as araby
import subprocess

import soundfile as sf


from artst.tasks.artst import ArTSTTask
from artst.models.artst import ArTSTTransformerModel
from fairseq.tasks.hubert_pretraining import LabelEncoder 

from fairseq import checkpoint_utils, options, scoring, tasks, utils

from loguru import logger
from fairseq.logging.meters import StopwatchMeter, TimeMeter


def postprocess(wav, cur_sample_rate):
    if wav.dim() == 2:
        wav = wav.mean(-1)
    assert wav.dim() == 1, wav.dim()

    if cur_sample_rate != 16000:
        raise Exception(f"sr {cur_sample_rate} != {16000}")
    return wav


def main(cfg: DictConfig, audio_path):
    
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout, audio_path)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file, audio_path):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    # task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    
    wav, cur_sample_rate = sf.read(audio_path)
    wav = torch.from_numpy(wav).float()
    wav = postprocess(wav, cur_sample_rate)
    sample = {'index': 0, 'net_input': {'source': torch.tensor(wav).unsqueeze(dim=0), 'padding_mask':torch.BoolTensor(wav.shape).fill_(False).unsqueeze(dim=0)}, 'id': [0], 'target': [[None], ]}

    prefix_tokens = None
    if cfg.generation.prefix_size > 0:
        prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

    constraints = None
    if "constraints" in sample:
        constraints = sample["constraints"]

    gen_timer.start()
    hypos = task.inference_step(
        generator,
        models,
        sample,
        prefix_tokens=prefix_tokens,
        constraints=constraints,
    )
    num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
    gen_timer.stop(num_generated_tokens)

    for i, sample_id in enumerate(sample["id"]):
        has_target = False

        # Remove padding
        if "src_tokens" in sample["net_input"]:
            src_tokens = utils.strip_pad(
                sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
            )
        else:
            src_tokens = None

        target_tokens = None
        if has_target:
            target_tokens = (
                utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
            )

        # Either retrieve the original sentences or regenerate them from tokens.
        if align_dict is not None:
            src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                sample_id
            )
            target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                sample_id
            )
        else:
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
            else:
                src_str = ""
            if has_target:
                target_str = tgt_dict.string(
                    target_tokens,
                    cfg.common_eval.post_process,
                    escape_unk=True,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                        generator
                    ),
                )

        src_str = decode_fn(src_str)
        if has_target:
            target_str = decode_fn(target_str)

        # Process top predictions
        for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=hypo["alignment"],
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=cfg.common_eval.post_process,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )
            detok_hypo_str = decode_fn(hypo_str)
            
    return detok_hypo_str
  
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    parser.add_argument('--data', type=str, default='/artst/utils', metavar='data')
    parser.add_argument('--bpe-tokenizer', type=str, default='/l/users/speech_lab/_SpeechT5PretrainDataset/FinetuneV2/v2.model')
    parser.add_argument('--user-dir', type=str, default='/l/users/speech_lab/_SpeechT5PretrainDataset/SpeechT5_inference/SpeechT5/speecht5')
    parser.add_argument('--task', type=str, default='artst')
    parser.add_argument('--t5-task', type=str, default='s2t')
    parser.add_argument('--path', type=str, default='/l/users/speech_lab/_SpeechT5PretrainDataset/Finetune/ASR/_models/v2spmMGB2/checkpoint_best.pt')
    parser.add_argument('--ctc-weight', type=float, default=0.25)
    parser.add_argument('--max-tokens', type=int, default=350000)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--scoring', type=str, default='wer')
    parser.add_argument('--max-len-a', type=float, default=0)
    parser.add_argument('--max-len-b', type=int, default=1000)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--batch-size', type=int, default=1)
    # parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument("-i",'--input_file', type=str, default='/input.tsv')
    parser.add_argument("-o",'--out_file', type=str, default='/output.tsv')

    args = parser.parse_args()
    
    
    with open(args.input_file, 'r') as f:
        for ind ,line in enumerate(f):
            audio, _ = line.strip().split('\t')
            try:
                transcript = main(args, audio_path=audio)
                out = open(args.out_file, 'a')
                print(f"{audio}\t{transcript}", file=out)
                out.close()
            except:
                print(f"Error processing {audio}")
