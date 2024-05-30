import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn

import sys
sys.path.append('/l/users/speech_lab/_DeepFakeDetection/Vocoder/ParallelWaveGAN/egs/clartts/convert')
from config_other_vocoders import StyleMelGanConfig

# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""StyleMelGAN's TADEResBlock Modules."""

from functools import partial

import torch


class TADELayer(torch.nn.Module):
    """TADE Layer module."""

    def __init__(
        self,
        in_channels=64,
        aux_channels=80,
        kernel_size=9,
        bias=True,
        upsample_factor=2,
        upsample_mode="nearest",
    ):
        """Initilize TADE layer."""
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(in_channels)
        self.aux_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                aux_channels,
                in_channels,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.gated_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                in_channels * 2,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )

    def _init_weights(self, initializer_range):
        self.aux_conv[0].weight.data.normal_(mean=0.0, std=initializer_range)
        self.gated_conv[0].weight.data.normal_(mean=0.0, std=initializer_range)

        if self.aux_conv[0].bias is not None:
            self.aux_conv[0].bias.data.zero_()
        if self.gated_conv[0].bias is not None:
            self.gated_conv[0].bias.data.zero_()

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').

        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled aux tensor (B, in_channels, T * aux_upsample_factor).

        """
        x = self.norm(x)
        c = self.upsample(c)
        c = self.aux_conv(c)
        cg = self.gated_conv(c)
        cg1, cg2 = cg.split(cg.size(1) // 2, dim=1)
        # NOTE(kan-bayashi): Use upsample for noise input here?
        y = cg1 * self.upsample(x) + cg2
        # NOTE(kan-bayashi): Return upsampled aux here?
        return y, c


class TADEResBlock(torch.nn.Module):
    """TADEResBlock module."""

    def __init__(
        self,
        in_channels=64,
        aux_channels=80,
        kernel_size=9,
        dilation=2,
        bias=True,
        upsample_factor=2,
        upsample_mode="nearest",
        gated_function="softmax",
    ):
        """Initialize TADEResBlock module."""
        super().__init__()
        self.tade1 = TADELayer(
            in_channels=in_channels,
            aux_channels=aux_channels,
            kernel_size=kernel_size,
            bias=bias,
            # NOTE(kan-bayashi): Use upsample in the first TADE layer?
            upsample_factor=1,
            upsample_mode=upsample_mode,
        )
        self.gated_conv1 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )
        self.tade2 = TADELayer(
            in_channels=in_channels,
            aux_channels=in_channels,
            kernel_size=kernel_size,
            bias=bias,
            upsample_factor=upsample_factor,
            upsample_mode=upsample_mode,
        )
        self.gated_conv2 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            dilation=dilation,
            padding=(kernel_size - 1) // 2 * dilation,
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )
        if gated_function == "softmax":
            self.gated_function = partial(torch.softmax, dim=1)
        elif gated_function == "sigmoid":
            self.gated_function = torch.sigmoid
        else:
            raise ValueError(f"{gated_function} is not supported.")

    def _init_weights(self, initializer_range):
        self.gated_conv1.weight.data.normal_(mean=0.0, std=initializer_range)
        self.gated_conv2.weight.data.normal_(mean=0.0, std=initializer_range)

        if self.gated_conv1.bias is not None:
            self.gated_conv1.bias.data.zero_()
        if self.gated_conv2.bias is not None:
            self.gated_conv2.bias.data.zero_()

        self.tade1._init_weights(initializer_range)
        self.tade2._init_weights(initializer_range)


    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').

        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled auxirialy tensor (B, in_channels, T * in_upsample_factor).

        """
        residual = x

        x, c = self.tade1(x, c)
        x = self.gated_conv1(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = self.gated_function(xa) * torch.tanh(xb)

        x, c = self.tade2(x, c)
        x = self.gated_conv2(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = self.gated_function(xa) * torch.tanh(xb)

        # NOTE(kan-bayashi): Return upsampled aux here?
        return self.upsample(residual) + x, c


class StyleMelGANGenerator(PreTrainedModel):
    """Style MelGAN generator module."""
    config_class=StyleMelGanConfig
    main_input_name = "spectrogram"

    def __init__(
        self,
        config
    ):
        """Initilize Style MelGAN generator.

        Args:
            in_channels (int): Number of input noise channels.
            aux_channels (int): Number of auxiliary input channels.
            channels (int): Number of channels for conv layer.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of conv layers.
            dilation (int): Dilation factor for conv layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            noise_upsample_scales (list): List of noise upsampling scales.
            noise_upsample_activation (str): Activation function module name for noise upsampling.
            noise_upsample_activation_params (dict): Hyperparameters for the above activation function.
            upsample_scales (list): List of upsampling scales.
            upsample_mode (str): Upsampling mode in TADE layer.
            gated_function (str): Gated function in TADEResBlock ("softmax" or "sigmoid").
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__(config)
        in_channels=128
        aux_channels=80
        channels=64
        out_channels=1
        kernel_size=9
        dilation=2
        bias=True
        noise_upsample_scales=[10, 2, 2, 2]
        noise_upsample_activation="LeakyReLU"
        noise_upsample_activation_params={"negative_slope": 0.2}
        upsample_scales=[4, 1, 4, 1, 4, 1, 2, 2, 1]
        upsample_mode="nearest"
        gated_function="softmax"
        use_weight_norm=True

        self.in_channels = in_channels
        self.initializer_range = 0.01

        noise_upsample = []
        in_chs = in_channels
        for noise_upsample_scale in noise_upsample_scales:
            # NOTE(kan-bayashi): How should we design noise upsampling part?
            noise_upsample += [
                torch.nn.ConvTranspose1d(
                    in_chs,
                    channels,
                    noise_upsample_scale * 2,
                    stride=noise_upsample_scale,
                    padding=noise_upsample_scale // 2 + noise_upsample_scale % 2,
                    output_padding=noise_upsample_scale % 2,
                    bias=bias,
                )
            ]
            noise_upsample += [
                getattr(torch.nn, noise_upsample_activation)(
                    **noise_upsample_activation_params
                )
            ]
            in_chs = channels
        self.noise_upsample = torch.nn.Sequential(*noise_upsample)
        self.noise_upsample_factor = np.prod(noise_upsample_scales)

        self.blocks = torch.nn.ModuleList()
        aux_chs = aux_channels
        for upsample_scale in upsample_scales:
            self.blocks += [
                TADEResBlock(
                    in_channels=channels,
                    aux_channels=aux_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=bias,
                    upsample_factor=upsample_scale,
                    upsample_mode=upsample_mode,
                    gated_function=gated_function,
                ),
            ]
            aux_chs = channels
        self.upsample_factor = np.prod(upsample_scales)

        self.output_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                channels,
                out_channels,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )

        self.register_buffer("mean", torch.zeros(80))
        self.register_buffer("scale", torch.ones(80))


        # Initialize weights and apply final processing
        self.post_init()


    def _init_weights(self, module):
        """Initialize the weights."""

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, TADEResBlock):
            module._init_weights(self.config.initializer_range)
        # elif isinstance(module, (ConvInUpsampleNetwork, MelGANGenerator)):
        #     module._init_weights()

        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            for sub_module in module:
                self._init_weights(sub_module)



    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


    def forward(self, c, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = c.transpose(1, 0).unsqueeze(0)

        # prepare noise input
        noise_size = (
            1,
            self.in_channels,
            (c.size(2) - 1) // self.noise_upsample_factor + 1,
        )
        noise = torch.randn(*noise_size, dtype=torch.float).to(
            next(self.parameters()).device
        )
        x = self.noise_upsample(noise)

        # NOTE(kan-bayashi): To remove pop noise at the end of audio, perform padding
        #    for feature sequence and after generation cut the generated audio. This
        #    requires additional computation but it can prevent pop noise.
        total_length = c.size(2) * self.upsample_factor
        c = F.pad(c, (0, x.size(2) - c.size(2)), "replicate")

        # This version causes pop noise.
        # x = x[:, :, :c.size(2)]

        for block in self.blocks:
            x, c = block(x, c)
        x = self.output_conv(x)[..., :total_length]

        is_batched = c.dim() == 3
        if not is_batched:
            # remove batch dim and collapse tensor to 1-d audio waveform
            hidden_states = x.squeeze(0).transpose(1, 0).view(-1)
        else:
            # remove seq-len dim since this collapses to 1
            hidden_states = x.squeeze(0)

        return hidden_states