import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
Convolutional Dense Encoder-Decoder Networks

Input --> Conv --> DownSampling --> DenseBlock --> Downsampling --------
                                                                        |
Output <-- Upsampling <-- DenseBlock <-- Upsampling <-- DenseBlock <----

"""
class UpsamplingNearest2d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
    

class UpsamplingBilinear2d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, 
            mode='bilinear', align_corners=True)



# class DenseResidualBlock(nn.Module):
#     """
#     The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
#     """

#     def __init__(self, filters, res_scale=0.2):
#         super(DenseResidualBlock, self).__init__()
#         self.res_scale = res_scale

#         def block(in_features, non_linearity=True):
#             layers = [nn.BatchNorm2d(in_features)]
#             layers += [nn.ReLU(inplace=True)]
#             layers += [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
#             return nn.Sequential(*layers)

#         self.b1 = block(in_features=1 * filters)
#         self.b2 = block(in_features=2 * filters)
#         self.b3 = block(in_features=3 * filters)
#         self.b4 = block(in_features=4 * filters)
#         self.b5 = block(in_features=5 * filters, non_linearity=False)
#         self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

#     def forward(self, x):
#         inputs = x
#         for block in self.blocks:
#             out = block(inputs)
#             inputs = torch.cat([inputs, out], 1)
#         return out.mul(self.res_scale) + x


# class ResidualInResidualDenseBlock(nn.Module):
#     def __init__(self, filters, res_scale=0.2):
#         super(ResidualInResidualDenseBlock, self).__init__()
#         self.res_scale = res_scale
#         self.dense_blocks = nn.Sequential(
#             DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)#, DenseResidualBlock(filters)
#         )

#     def forward(self, x):
#         return self.dense_blocks(x).mul(self.res_scale) + x



class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.
    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): dropout rate
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    # bottleneck layer, bn_size: bottleneck size
    def __init__(self, in_features, growth_rate, drop_rate=0, bn_size=4,
                 bottleneck=False):
        # detect if the input features are more than bn_size x k,
        # if yes, use bottleneck -- not much memory gain, but lose one relu
        # I disabled the bottleneck for current implementation
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        # Add dropout in last layer if needed
        if self.drop_rate > 0:
            y = F.dropout2d(y, p=self.drop_rate, training=self.training)
        z = torch.cat([x, y], 1)
        return z


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    """Transition layer, either downsampling or upsampling, both reduce
    number of feature maps, i.e. `out_features` should be less than 
    `in_features`.
    Args:
        in_features (int):
        out_features (int):
        down (bool): If True, downsampling, else upsampling
        bottleneck (bool, True): If True, enable bottleneck design
        drop_rate (float, 0.):
    """
    
    def __init__(self, in_features, out_features, encoding=True, bottleneck=True, drop_rate=0.,
                 last=False, out_channels=3, outsize_even=True, upsample=None):
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if encoding:
            # half feature resolution, reduce # feature maps
            if bottleneck == False:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                # Add dropout if needed
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
                # not using pooling, fully convolutional...
                self.add_module('conv2', nn.Conv2d(out_features, out_features,
                    kernel_size=3, stride=2, 
                    padding=1, bias=False))
                # Add dropout if needed
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=2, 
                    padding=0, bias=False))
                # Add dropout if needed
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
        else:
            # transition up, increase feature resolution, half # feature maps
            if bottleneck == False:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # output_padding=0, or 1 depends on the image size
                # if image size is of the power of 2, then 1 is good
                if upsample is None:
                    self.add_module('convT2', nn.ConvTranspose2d(
                        out_features, out_features, kernel_size=3, stride=2,
                        padding=1, output_padding=1, bias=False))
                elif upsample == 'linear':
                    self.add_module('upsample', UpsamplingBilinear2d(scale_factor=2))
                    self.add_module('conv2', nn.Conv2d(out_features, out_features,
                        3, 1, 1*2, bias=False, padding_mode='circular'))
                elif upsample == 'nearest':
                    self.add_module('upsample', UpsamplingNearest2d(scale_factor=2))
                    self.add_module('conv2', nn.Conv2d(out_features, out_features,
                        3, 1, 1*2, bias=False, padding_mode='circular'))

            else:
                if upsample is None:
                    self.add_module('convT2', nn.ConvTranspose2d(
                        in_features, out_features, kernel_size=3, stride=2,
                        padding=1, output_padding=1, bias=False))
                elif upsample == 'linear':
                    self.add_module('upsample', UpsamplingBilinear2d(scale_factor=2))
                    self.add_module('conv2', nn.Conv2d(in_features, out_features,
                        3, 1, 1*2, bias=False, padding_mode='circular'))
                elif upsample == 'nearest':
                    self.add_module('upsample', UpsamplingNearest2d(scale_factor=2))
                    self.add_module('conv2', nn.Conv2d(in_features, out_features,
                        3, 1, 1*2, bias=False, padding_mode='circular'))

            if drop_rate > 0:
                self.add_module('dropout1', nn.Dropout2d(p=drop_rate))


def last_decoding(in_features, out_channels, outsize_even=True, drop_rate=0., upsample=None):
    """Last transition up layer, which outputs directly the predictions.
    """
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm2d(in_features))
    last_up.add_module('relu1', nn.ReLU(True))
    last_up.add_module('conv1', nn.Conv2d(in_features, in_features // 2, 
                    kernel_size=1, stride=1, padding=0, bias=False))
    # Add dropout if needed
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout2d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm2d(in_features // 2))
    last_up.add_module('relu2', nn.ReLU(True))
    
    # output layer
    if upsample is None:
        ks = 6 if outsize_even else 5
        last_up.add_module('convT2', nn.ConvTranspose2d(in_features // 2, out_channels,
                        kernel_size=ks, stride=2, padding=2, bias=False))
    elif upsample == 'nearest':
        last_up.add_module('upsample', UpsamplingNearest2d(scale_factor=2))
    elif upsample == 'linear':
        last_up.add_module('upsample', UpsamplingBilinear2d(scale_factor=2))
    
    return last_up


class DenseED(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                 num_init_features=64, bn_size=4, drop_rate=0, outsize_even=False,
                 bottleneck=False,upsample=None,out_activation=None):
        """
        In the network presented in the paper, the last decoding layer 
        (transition up) directly outputs the predicted fields. 
        The network parameters should be modified for different image size,
        mostly the first conv and the last convT layers. (`output_padding` in
        ConvT can be modified as well)
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            blocks: list (of odd size) of integers
            growth_rate (int): K
            num_init_features (int): the number of feature maps after the first
                conv layer
            bn_size: bottleneck size for number of feature maps (not useful...)
            bottleneck (bool): use bottleneck for dense block or not
            drop_rate (float): dropout rate
            outsize_even (bool): if the output size is even or odd (e.g.
                65 x 65 is odd, 64 x 64 is even)
            out_activation: Output activation function, choices=[None, 'tanh',
                'sigmoid', 'softplus']
            upsample: How to upsample in the encoding phase choices = [None(Conv2T), nearest, linear]

        """
        super(DenseED, self).__init__()
        self.out_channels = out_channels

        if len(blocks) > 1 and len(blocks) % 2 == 0:
            ValueError('length of blocks must be an odd number, but got {}'
                       .format(len(blocks)))
        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]
        self.features = nn.Sequential()
        ' First convolution ================'
        # only conv, half image size
        # For even image size: k7s2p3, k5s2p2
        # For odd image size (e.g. 65): k7s2p2, k5s2p1, k13s2p5, k11s2p4, k9s2p3
        self.features.add_module('in_conv',
                    nn.Conv2d(in_channels, num_init_features,
                            kernel_size=7, stride=2, padding=3, bias=False))

        # Encoding / transition down ================
        # dense block --> encoding --> dense block --> encoding
        num_features = num_init_features
        for i, num_layers in enumerate(enc_block_layers):
            'After First Convolution --> Dense Block' 
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            
            self.features.add_module('encblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            'Encoding layer'
            trans = _Transition(in_features=num_features,
                                out_features=num_features // 2,
                                encoding=True, 
                                drop_rate=drop_rate,
                                bottleneck=bottleneck)
            self.features.add_module('down%d' % (i + 1), trans)
            num_features = num_features // 2

        ' Decoding / transition up =============='
        # dense block --> decoding --> dense block --> decoding --> dense block
        # if len(dec_block_layers) - len(enc_block_layers) == 1:
        for i, num_layers in enumerate(dec_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            self.features.add_module('decblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate

            # the last decoding layer has different convT parameters
            if i < len(dec_block_layers) - 1:
                trans = _Transition(in_features=num_features,
                                    out_features=num_features // 2,
                                    encoding=False,
                                    bottleneck=bottleneck,
                                    drop_rate=drop_rate,
                                    out_channels=out_channels,
                                    outsize_even=outsize_even,
                                    upsample=upsample)
                self.features.add_module('up%d' % (i + 1), trans)
                num_features = num_features // 2

        # The last decoding layer =======
        last_trans_up = last_decoding(num_features, out_channels,outsize_even=outsize_even, drop_rate=0.,
                                upsample=upsample)
        self.features.add_module('LastTransUp', last_trans_up)
    
    # forward the sequential model
    def forward(self, x):
        y = self.features(x)

        # use the softplus activation for concentration (always positive)
        y = torch.sigmoid(y)#F.softplus(y.clone(), beta=1)

        return y
    # get the total number of parameters of the convolutional layers network
    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
