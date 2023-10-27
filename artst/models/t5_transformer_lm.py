# --------------------------------------------------------
# ArTST: Arabic Text and Speech Transformer (https://arxiv.org/abs/2310.16621)
# Github source: https://github.com/mbzuai-nlp/ArTST
# Based on speecht5, fairseq and espnet code bases
# https://github.com/microsoft/SpeechT5/tree/main/SpeechT5; https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

from fairseq.models import (
    register_model_architecture,
)
from fairseq.models.transformer_lm import base_lm_architecture


# @register_model_architecture(model_name="transformer_lm", arch_name="transformer_lm_t5")
def transformer_lm_t5(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1280)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 6144)
    args.decoder_layers = getattr(args, "decoder_layers", 20)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)
