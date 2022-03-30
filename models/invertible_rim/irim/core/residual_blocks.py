from torch import nn

from models.invertible_rim.irim.utils.torch_utils import determine_conv_class


class ResidualBlockPixelshuffle(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, kernel_size=3, dilation=1, conv_nd=2, use_glu=True):
        super(ResidualBlockPixelshuffle, self).__init__()
        self.n_output = n_output
        self.conv_nd = conv_nd
        self.use_glu = use_glu
        conv_layer = determine_conv_class(conv_nd, transposed=False)
        transposed_conv_layer = determine_conv_class(conv_nd, transposed=True)

        if use_glu:
            n_output = n_output * 2

        self.l1 = nn.utils.weight_norm(conv_layer(n_input, n_hidden, kernel_size=dilation,
                                                  stride=dilation, padding=0, bias=True))
        self.l2 = nn.utils.weight_norm(conv_layer(n_hidden, n_hidden, kernel_size=kernel_size,
                                                  padding=kernel_size // 2, dilation=1, bias=True))
        self.l3 = nn.utils.weight_norm(transposed_conv_layer(n_hidden, n_output, kernel_size=dilation,
                                                             stride=dilation, padding=0, bias=False))

    def forward(self, x):
        x_size = list(x.size())
        x_size[1] = self.n_output

        x = nn.functional.relu(self.l1(x))
        x = nn.functional.relu(self.l2(x))
        x = self.l3(x, output_size=tuple(x_size))
        if self.use_glu:
            x = nn.functional.glu(x, 1)

        return x
