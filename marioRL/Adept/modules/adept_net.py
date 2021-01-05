"""
    Custom adept network
    Inherit from adept/network/net3d/four_conv
"""
from torch import nn
from torch.nn import functional as F
from adept.scripts.local import parse_args, main
from adept.network import SubModule3D

class AdeptMarioNet(SubModule3D):
    # You will be prompted for these when training script starts
    args = {}

    # in_shape = input_dim, id = name of the network if you want to give to it
    def __init__(self, in_shape, id):
        super(AdeptMarioNet, self).__init__(in_shape, id)
        # Set properties and whatnot here
        c, h, w = in_shape

        self._in_shape = in_shape
        self._out_shape = None

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # Conversion from sequential model
        # Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32,
                               kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1)
        # Batch Normalization # equal number of out_channel - just better for training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # Set the weights
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, args, in_shape, id):
        return cls(in_shape, id)

    @property
    def _output_shape(self):
        # For 84x84, (32, 5, 5)
        # Call helper function
        if self._out_shape is None: # what does this mean? kernel size, stride, padding, dilation
            output_dim = calc_output_dim(self._in_shape[1], 8, 4, 0, 1)
            output_dim = calc_output_dim(output_dim, 4, 2, 0, 1)
            output_dim = calc_output_dim(output_dim, 3, 1, 0, 1)
            self._out_shape = 64, output_dim, output_dim
        return self._out_shape

    # Three layers convolution
    def _forward(self, xs, internals, **kwargs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        return xs, {}

    def _new_internals(self):
        return {}


def calc_output_dim(dim_size, kernel_size, stride, padding, dilation):
    numerator = dim_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return numerator // stride + 1

