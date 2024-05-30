import copy
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn

import sys
sys.path.append('/l/users/speech_lab/_DeepFakeDetection/Vocoder/ParallelWaveGAN/egs/clartts/convert')
from config_other_vocoders import ParallelWaveGanConfig


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def _init_weights(self, initializer_range):
        self.weight.data.normal_(mean=0.0, std=initializer_range)
        if self.bias is not None:
            self.bias.data.zero_()

    

class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )

class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode="nearest"):
        """Initialize Stretch2d module.

        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            mode (str): Interpolation mode.

        """
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, C, F, T).

        Returns:
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale),

        """
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode
        )

class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(
        self,
        upsample_scales,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode="nearest",
        freq_axis_kernel_size=1,
        use_causal_conv=False,
    ):
        """Initialize upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            interpolate_mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.

        """
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        self.upsample_scales = upsample_scales
        for scale in upsample_scales:
            # interpolation layer
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]

            # conv layer
            assert (
                freq_axis_kernel_size - 1
            ) % 2 == 0, "Not support even number freq axis kernel size."
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            if use_causal_conv:
                padding = (freq_axis_padding, scale * 2)
            else:
                padding = (freq_axis_padding, scale)
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]

            # nonlinear
            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params
                )
                self.up_layers += [nonlinear]

    def _init_weights(self, initializer_range):
        for i in range(len(self.upsample_scales)):
            if i % 2 == 1:
                self.up_layers[i].weight.data.normal_(mean=0.0, std=initializer_range)
                if self.up_layers[i].bias is not None:
                    self.up_layers[i].bias.data.zero_()

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T).

        Returns:
            Tensor: Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).

        """
        c = c.unsqueeze(1)  # (B, 1, C, T)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, Conv2d):
                c = f(c)[..., : c.size(-1)]
            else:
                c = f(c)
        return c.squeeze(1)  # (B, C, T')



class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(
        self,
        upsample_scales,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode="nearest",
        freq_axis_kernel_size=1,
        aux_channels=80,
        aux_context_window=0,
        use_causal_conv=False,
    ):
        """Initialize convolution + upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            aux_channels (int): Number of channels of pre-convolutional layer.
            aux_context_window (int): Context window size of the pre-convolutional layer.
            use_causal_conv (bool): Whether to use causal structure.

        """
        super(ConvInUpsampleNetwork, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        # To capture wide-context information in conditional features
        kernel_size = (
            aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        )
        # NOTE(kan-bayashi): Here do not use padding because the input is already padded
        self.conv_in = Conv1d(
            aux_channels, aux_channels, kernel_size=kernel_size, bias=False
        )
        self.upsample = UpsampleNetwork(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv,
        )

    def _init_weights(self, initializer_range):
        self.conv_in.weight.data.normal_(mean=0.0, std=initializer_range)
        self.upsample._init_weights(initializer_range)

        if self.conv_in.bias is not None:
            self.conv_in.bias.data.zero_()


    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T').

        Returns:
            Tensor: Upsampled tensor (B, C, T),
                where T = (T' - aux_context_window * 2) * prod(upsample_scales).

        Note:
            The length of inputs considers the context window size.

        """
        c_ = self.conv_in(c)
        c = c_[:, :, : -self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(
        self,
        kernel_size=3,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        dropout=0.0,
        dilation=1,
        bias=True,
        use_causal_conv=False,
    ):
        """Initialize WaveNetResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dropout (float): Dropout probability.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_causal_conv (bool): Whether to use use_causal_conv or non-use_causal_conv convolution.

        """
        super().__init__()
        self.dropout = dropout
        # no future time stamps available
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv

        # dilation conv
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def _init_weights(self, initializer_range):
        """Initialize the weights."""
        self.conv.weight.data.normal_(mean=0.0, std=initializer_range)
        self.conv1x1_aux.weight.data.normal_(mean=0.0, std=initializer_range)
        self.conv1x1_out.weight.data.normal_(mean=0.0, std=initializer_range)
        self.conv1x1_skip.weight.data.normal_(mean=0.0, std=initializer_range)

        if self.conv.bias is not None:
            self.conv.bias.data.zero_()
        if self.conv1x1_aux.bias is not None:
            self.conv1x1_aux.bias.data.zero_()
        if self.conv1x1_out.bias is not None:
            self.conv1x1_out.bias.data.zero_()
        if self.conv1x1_skip.bias is not None:
            self.conv1x1_skip.bias.data.zero_()


    def forward(self, x, spectrogram):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            spectrogram (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)

        # remove future time steps if use_causal_conv conv
        x = x[:, :, : residual.size(-1)] if self.use_causal_conv else x

        # split into two part for gated activation
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if spectrogram is not None:
            assert self.conv1x1_aux is not None
            spectrogram = self.conv1x1_aux(spectrogram)
            ca, cb = spectrogram.split(spectrogram.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

        return x, s



class ParallelWaveGANGenerator(PreTrainedModel):
    config_class = ParallelWaveGanConfig
    main_input_name = "spectrogram"
    """Parallel WaveGAN Generator module."""

    def __init__(
        self,
        config
    ):
        """Initialize Parallel WaveGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.

        """
        in_channels=1
        out_channels=1
        kernel_size=3
        layers=30
        stacks=3
        residual_channels=64
        gate_channels=128
        skip_channels=64
        aux_channels=80
        aux_context_window=2
        dropout=0.0
        bias=True
        use_weight_norm=True
        use_causal_conv=False
        upsample_conditional_features=True
        upsample_net="ConvInUpsampleNetwork"
        upsample_params={"upsample_scales": [4, 4, 4, 4]}
        super(ParallelWaveGANGenerator, self).__init__(config)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.initializer_range = 0.01

        # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define first convolution
        self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

        # define conv + upsampling network
        if upsample_conditional_features:
            upsample_params.update(
                {
                    "use_causal_conv": use_causal_conv,
                }
            )
            if upsample_net == "MelGANGenerator":
                assert aux_context_window == 0
                upsample_params.update(
                    {
                        "use_weight_norm": False,  # not to apply twice
                        "use_final_nonlinear_activation": False,
                    }
                )
                self.upsample_net = getattr(models, upsample_net)(**upsample_params)
            else:
                if upsample_net == "ConvInUpsampleNetwork":
                    upsample_params.update(
                        {
                            "aux_channels": aux_channels,
                            "aux_context_window": aux_context_window,
                        }
                    )
                self.upsample_net = ConvInUpsampleNetwork(**upsample_params)
            self.upsample_factor = np.prod(upsample_params["upsample_scales"])
        else:
            self.upsample_net = None
            self.upsample_factor = 1

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=use_causal_conv,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            ]
        )
        
        self.register_buffer("mean", torch.zeros(80))
        self.register_buffer("scale", torch.ones(80))
        # apply weight norm
        self.post_init()


    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, (ResidualBlock, ConvInUpsampleNetwork)):
            module._init_weights(self.config.initializer_range)
        
        if isinstance(module, Conv1d1x1):
            module._init_weights(self.config.initializer_range)

        if isinstance(module, nn.ModuleList):
            for sub_module in module:
                self._init_weights(sub_module)

    def _forward(self, z, spectrogram):
        """Calculate forward propagation.

        Args:
            z (Tensor): Input noise signal (B, 1, T).
            spectrogram (Tensor): Local conditioning auxiliary features (B, spectrogram ,T').

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        # perform upsampling
        if spectrogram is not None and self.upsample_net is not None:
            spectrogram = self.upsample_net(spectrogram)
            assert spectrogram.size(-1) == z.size(-1)

        # encode to hidden representation
        x = self.first_conv(z)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, spectrogram)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        is_batched = spectrogram.dim() == 3
        if not is_batched:
            # remove batch dim and collapse tensor to 1-d audio waveform
            hidden_states = x.squeeze(0).transpose(1, 0).view(-1)
        else:
            # remove seq-len dim since this collapses to 1
            hidden_states = x.squeeze(0)

        return hidden_states


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
        'apply'
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, Conv1d1x1):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


    def forward(self, spectrogram=None, x=None, normalize_before=False):
        """Perform inference.

        Args:
            spectrogram (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,spectrogram).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T, out_channels)

        """
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert spectrogram is not None
            x = torch.randn(1, 1, len(spectrogram) * self.upsample_factor).to(
                next(self.parameters()).device
            )
        if spectrogram is not None:
            if not isinstance(spectrogram, torch.Tensor):
                spectrogram = torch.tensor(spectrogram, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            if normalize_before:
                spectrogram = (spectrogram - self.mean) / self.scale
            spectrogram = spectrogram.transpose(1, 0).unsqueeze(0)
            spectrogram = torch.nn.ReplicationPad1d(self.aux_context_window)(spectrogram)
        return self._forward(x, spectrogram)


