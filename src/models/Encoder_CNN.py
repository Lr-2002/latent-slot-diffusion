from torch import nn
import torch
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class BatchNormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, res=False):
        super(BatchNormConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = res

    def forward(self, x):
        if self.residual:
            return x + self.relu(self.bn1(self.conv1(x)))
        else:
            return self.relu(self.bn1(self.conv1(x)))


class BatchNormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, res=False):
        super(BatchNormConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = res
    def forward(self, x):
        if self.residual:
            return x+self.relu(self.bn1(self.conv1(x)))
        else:
            return self.relu(self.bn1(self.conv1(x)))


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)


class EncoderCNN( ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, cnn_hidden_size, d_model):
        super().__init__()
        self.object_encoder_cnn = nn.Sequential(
            BatchNormConvBlock(cnn_hidden_size, cnn_hidden_size, 7, 3, 0),
            BatchNormConvBlock(cnn_hidden_size, cnn_hidden_size, 7, 3, 0),
            nn.Conv2d(cnn_hidden_size, d_model, 5, 1, 0),
        )
        self.mlp = nn.Sequential(
            linear(d_model, d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(d_model, d_model))

        self.spatial_pos = CartesianPositionalEmbedding(cnn_hidden_size, 64)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.object_encoder_cnn(x)