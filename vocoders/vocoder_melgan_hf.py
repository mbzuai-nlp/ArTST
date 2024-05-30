import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn

import sys
sys.path.append('/l/users/speech_lab/_DeepFakeDetection/Vocoder/ParallelWaveGAN/egs/clartts/convert')
from config_other_vocoders import MelGanConfig

"""Residual stack module in MelGAN."""
class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(
        self,
        kernel_size=3,
        channels=32,
        dilation=1,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_causal_conv=False,
    ):
        super(ResidualStack, self).__init__()

        # defile residual stack part
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        self.stack = torch.nn.Sequential(
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params),
            torch.nn.Conv1d(
                channels, channels, kernel_size, dilation=dilation, bias=bias
            ),
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            torch.nn.Conv1d(channels, channels, 1, bias=bias),
        )
        # defile extra layer for skip connection
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        return self.stack(c) + self.skip_layer(c)


    # def apply_weight_norm(self):
    #     nn.utils.weight_norm(self.stack[2])  # conv1
    #     nn.utils.weight_norm(self.stack[4])  # conv1
    #     nn.utils.weight_norm(self.skip_layer)

    # def remove_weight_norm(self):
    #     nn.utils.remove_weight_norm(self.stack[2])  # conv1
    #     nn.utils.remove_weight_norm(self.stack[4])  # conv1
    #     nn.utils.remove_weight_norm(self.skip_layer)

    def _init_weights(self, initializer_range):
        """Initialize the weights."""
        self.stack[2].weight.data.normal_(mean=0.0, std=initializer_range)
        self.stack[4].weight.data.normal_(mean=0.0, std=initializer_range)
        self.skip_layer.weight.data.normal_(mean=0.0, std=initializer_range)

        if self.stack[2].bias is not None:
            self.stack[2].bias.data.zero_()
        if self.stack[4].bias is not None:
            self.stack[4].bias.data.zero_()
        if self.skip_layer.bias is not None:
            self.skip_layer.bias.data.zero_()



class MelGANGenerator(PreTrainedModel):
    """MelGAN generator module."""
    config_class = MelGanConfig
    main_input_name = "spectrogram"

    def __init__(
        self,
        config
    ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        in_channels=80
        out_channels=1
        kernel_size=7
        channels=512
        bias=True
        upsample_scales=[4, 4, 4, 4]
        stack_kernel_size=3
        stacks=3
        nonlinear_activation="LeakyReLU"
        nonlinear_activation_params={"negative_slope": 0.2}
        pad="ReflectionPad1d"
        pad_params={}
        use_final_nonlinear_activation=True
        use_weight_norm=True
        use_causal_conv=False
        super(MelGANGenerator, self).__init__(config)

        # check hyper parameters is valid
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 * len(upsample_scales)) == 0
        # add initial layer
        layers = []
        layers += [
            getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
            torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
        ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
            ]
            layers += [
                torch.nn.ConvTranspose1d(
                    channels // (2**i),
                    channels // (2 ** (i + 1)),
                    upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    bias=bias,
                )
            ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size**j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

        # add final layer
        layers += [
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        ]
        layers += [
            getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias
            ),
        ]

        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        self.register_buffer("mean", torch.zeros(80))
        self.register_buffer("scale", torch.ones(80))


        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""

        if getattr(module, "melgan", None):
            module = module.melgan
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, ResidualStack):
            module._init_weights(self.config.initializer_range)
        elif isinstance(module, nn.Sequential):
        # Iterate through modules in Sequential and initialize
            for sub_module in module:
                self._init_weights(sub_module)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                # if isinstance(m, ResidualStack):
                #     m.remove_weight_norm()
                # else:
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
            # if isinstance(m, ResidualStack):
            #     m.apply_weight_norm()

        self.apply(_apply_weight_norm)


    def forward(self, spectrogram: torch.FloatTensor, normalize_before: bool=True):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        is_batched = spectrogram.dim() == 3
        if normalize_before:
            spectrogram = (spectrogram - self.mean) / self.scale
        hidden_states = self.melgan(spectrogram.transpose(1, 0).unsqueeze(0))

        if not is_batched:
            # remove batch dim and collapse tensor to 1-d audio waveform
            hidden_states = hidden_states.squeeze(0).transpose(1, 0).view(-1)
        else:
            # remove seq-len dim since this collapses to 1
            hidden_states = hidden_states.squeeze(1)
        return hidden_states

