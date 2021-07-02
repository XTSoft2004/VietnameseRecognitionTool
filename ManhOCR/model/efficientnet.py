import math

import torch
from torch import nn
from torch.nn import functional as F

from ..model.utils import (
    MemoryEfficientSwish,
    Swish,
    calculate_output_image_size,
    drop_connect,
    efficientnet,
    get_same_padding_conv2d,
)


def round_repeats(repeats, depth_coefficient):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.

    Args:
        repeats (int): num_repeat to be calculated.

    Returns:
        :param repeats: New repeat number after calculating.
        :param depth_coefficient:
    """
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = (
            1 - global_params.batch_norm_momentum
        )  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = (
            block_args.id_skip
        )  # whether to use skip connection and drop connect
        self.stride = self._block_args.stride

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = (
            self._block_args.input_filters * self._block_args.expand_ratio
        )  # number of output channels

        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            stride=self.stride,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        image_size = calculate_output_image_size(image_size, self.stride)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if (
            self.id_skip
            and self._block_args.stride == 1
            and input_filters == output_filters
        ):
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class CustomEfficientNet(nn.Module):
    def __init__(self, image_size=None, drop_connect_rate=0.2):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        w, d, s, p = 1.0, 1.0, 224, 0.2

        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s
        )
        self._blocks_args, self._global_params = blocks_args, global_params
        image_size = image_size

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # out_channel = 48
        self._conv = Conv2d(
            in_channels=3, out_channels=48, kernel_size=3, stride=2, bias=False
        )
        self._swish = MemoryEfficientSwish()
        self._bn0 = nn.BatchNorm2d(num_features=48, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, stride=2)

        # Build blocks
        self._blocks = nn.ModuleList([])

        for i, block_args in enumerate(self._blocks_args):
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                num_repeat=round_repeats(
                    block_args.num_repeat, self._global_params.depth_coefficient
                )
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MBConvBlock(block_args, self._global_params, image_size=image_size)
            )
            image_size = calculate_output_image_size(image_size, block_args.stride)

            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )

            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(block_args, self._global_params, image_size=image_size)
                )

        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.conv1x1 = Conv2d(
            in_channels=448, out_channels=256, kernel_size=1, stride=1, bias=False
        )
        self._bn1 = nn.BatchNorm2d(num_features=256, momentum=bn_mom, eps=bn_eps)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn0(x)
        x = self._swish(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.drop_connect_rate

            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self.conv1x1(x)))
        x = x.transpose(-1, -2)
        x = x.flatten(2)
        x = x.permute(-1, 0, 1)

        return x


def efficient_net(image_size=(32, 356)):
    model = CustomEfficientNet(image_size=image_size)

    return model
