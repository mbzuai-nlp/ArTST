import ast
import logging
import os
import os.path as op
import sys
from argparse import Namespace

import numpy as np
import soundfile as sf
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from omegaconf import DictConfig


from transformers import SpeechT5HifiGan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(device)

# define function to calculate focus rate
# (see section 3.3 in https://arxiv.org/abs/1905.09263)
def _calculate_focus_rete(att_ws):
    if att_ws is None:
        # fastspeech case -> None
        return 1.0
    elif len(att_ws.shape) == 2:
        # tacotron 2 case -> (L, T)
        return float(att_ws.max(dim=-1)[0].mean())
    elif len(att_ws.shape) == 4:
        # transformer case -> (#layers, #heads, L, T)
        return float(att_ws.max(dim=-1)[0].mean(dim=-1).max())
    else:
        raise ValueError("att_ws should be 2 or 4 dimensional tensor.")


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)

    return _main(cfg, sys.stdout)


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("artst.generate_speech")

    utils.import_user_module(cfg.common)

    assert cfg.dataset.batch_size == 1, "only support batch size 1"
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if not use_cuda:
        logger.info("generate speech on cpu")

    # build task
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    logger.info(saved_cfg)

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=None,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )
    
    for i, sample in enumerate(progress):
        if "net_input" not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
    
        outs, _, attn = task.generate_speech(
            models, 
            sample["net_input"],
        )
        focus_rate = _calculate_focus_rete(attn)
        with torch.no_grad():
            gen_audio = vocoder(outs)

        gen_audio = gen_audio.cpu().numpy()
        
        outs = outs.cpu().numpy()
        audio_name = op.basename(sample['name'][0])
        if cfg.dataset.gen_subset == "zero_shot_test":
            audio_name = "_".join(sample['name'][0].split('/')[-3:])
       
        file_name = cfg.common_eval.results_path + audio_name
        sf.write(file_name, gen_audio, 16000)
        print("{} (size: {}->{} ({}), focus rate: {:.3f})".format(
                sample['name'][0],
                sample['src_lengths'][0].item(),
                outs.shape[0],
                sample['dec_target_lengths'][0].item(), 
                focus_rate
            ))
        


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
