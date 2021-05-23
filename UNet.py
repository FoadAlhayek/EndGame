from torch import nn
import torch
from torchsummary import summary


class ContractiveBlock(nn.Module):
    """
    Class defining the contraction blocks. Each contration block is
    composed of 2 conv. layers and a max pooling of stride 2x2 to cut the
    size in half. Each conv. layer is followed by a ReLU and batch norm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 pooling=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.pooling = pooling

        # Encoder
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               kernel_size=self.kernel_size, stride=1,
                               padding=self.padding)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=self.kernel_size, stride=1,
                               padding=self.padding)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(self.out_channels)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, y):
        x = self.conv1(y)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        before_pooling = x  # Save outputs so they can be skipped

        if self.pooling:    # Don't want to pool at the last block
            x = self.pool(x)

        return x, before_pooling


class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.up_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels,
                                          kernel_size=2, stride=2)
        self.relu0 = nn.ReLU()
        self.norm0 = nn.BatchNorm2d(self.out_channels)
        # self.concat = torch.cat((layer1, layer2), 1))
        self.conv1 = nn.Conv2d(2 * self.out_channels, self.out_channels,
                               kernel_size=self.kernel_size, stride=1,
                               padding=self.padding)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=self.kernel_size, stride=1,
                               padding=self.padding)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, encoder_layer, decoder_layer):
        up_layer = self.up_conv(decoder_layer)
        up_layer = self.relu0(up_layer)
        up_layer = self.norm0(up_layer)

        cat_layer = torch.cat((up_layer, encoder_layer), 1)
        x = self.conv1(cat_layer)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=4, start_filters=32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Encoder path
        for i in range(self.n_blocks):
            n_filters_in = self.in_channels if i == 0 else n_filters_out
            n_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.n_blocks - 1 else False  # No pool last block

            down_block = ContractiveBlock(in_channels=n_filters_in,
                                          out_channels=n_filters_out,
                                          kernel_size=3,
                                          padding=1,
                                          pooling=pooling)

            self.down_blocks.append(down_block)

        # Decoder path
        for i in range(self.n_blocks - 1):
            n_filters_in = n_filters_out
            n_filters_out = n_filters_in // 2

            up_block = ExpandingBlock(in_channels=n_filters_in,
                                      out_channels=n_filters_out,
                                      kernel_size=3,
                                      padding=1)

            self.up_blocks.append(up_block)

        self.conv_final = nn.Conv2d(n_filters_out, self.out_channels, kernel_size=3)

    def forward(self, x):
        encoder_outputs = []

        # Encoder path
        for d_block in self.down_blocks:
            x, before_pooling = d_block(x)
            encoder_outputs.append(before_pooling)

        # Decoding path
        i = 0
        for u_block in self.up_blocks:
            # Testa med (i+2)
            before_pooling = encoder_outputs[-(i + 2)]  # Due to appending, last is first
            x = u_block(before_pooling, x)
            i += 1

        # Last layer
        x = self.conv_final(x)

        return x


if __name__ == '__main__':
    # Check CUDA support
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create UNET example:
    model = UNet(in_channels=1,
                 out_channels=2,
                 n_blocks=4,
                 start_filters=32).to(device)

    summary(model, (1, 512, 512))
