import torch
import torch.nn as nn


def get_upscaling_block(
    channels_in, channels_out, kernel, stride, padding, last_layer=False
):
    """
    Each transpose conv will be followed by BatchNorm and ReLU,
    except the last block (which is only followed by tanh)
    """
    if last_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(
                channels_in, channels_out, kernel, stride, padding, bias=False
            ),
            nn.Tanh(),
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(
                channels_in, channels_out, kernel, stride, padding, bias=False
            ),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
        )


class Generator(nn.Module):
    def __init__(self, nz, ngf, nchannels=1):
        """
        nz: The latent size (100 in our case)
        ngf: The channel-size before the last layer (32 our case)
        """
        super().__init__()
        self.model = nn.Sequential(
            get_upscaling_block(nz, ngf * 4, 4, 1, 0),
            get_upscaling_block(ngf * 4, ngf * 2, 4, 2, 1),
            get_upscaling_block(ngf * 2, ngf, 4, 2, 1),
            get_upscaling_block(ngf, nchannels, 4, 2, 1, last_layer=True),
        )

    def forward(self, z):
        x = z.unsqueeze(2).unsqueeze(2)  # give spatial dimensions to z
        return self.model(x)


def get_downscaling_block(
    channels_in,
    channels_out,
    kernel,
    stride,
    padding,
    use_batch_norm=True,
    is_last=False,
):
    if is_last:
        return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
    elif use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.2),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel, stride, padding, bias=False),
            nn.LeakyReLU(0.2),
        )


class Discriminator(nn.Module):
    def __init__(self, ndf, nchannels=1):
        super().__init__()
        self.model = nn.Sequential(
            get_downscaling_block(nchannels, ndf, 4, 2, 1),
            get_downscaling_block(ndf, ndf * 2, 4, 2, 1),
            get_downscaling_block(ndf * 2, ndf * 4, 4, 2, 1),
            get_downscaling_block(ndf * 4, 1, 4, 1, 0, is_last=True),
        )

    def forward(self, x):
        return self.model(x).squeeze(1).squeeze(1)  # remove spatial dimensions


class ConditionalGenerator(nn.Module):
    def __init__(self, nz, nc, ngf, nchannels=1):
        super().__init__()
        self.upscaling_z = get_upscaling_block(nz, ngf * 8, 4, 1, 0)
        self.upscaling_c = get_upscaling_block(nc, ngf * 8, 4, 1, 0)
        self.rest_model = nn.Sequential(
            get_upscaling_block(ngf * 16, ngf * 8, 4, 2, 1),  # ngf*8 + ngf*8 channel
            get_upscaling_block(ngf * 8, ngf * 4, 4, 2, 1),
            get_upscaling_block(ngf * 4, nchannels, 4, 2, 1, last_layer=True),
        )

    def forward(self, x, y):
        x = x.unsqueeze(2).unsqueeze(2)
        x = self.upscaling_z(x)

        y = y.unsqueeze(2).unsqueeze(2)
        y = self.upscaling_c(y)

        x = torch.cat((x, y), dim=1)
        return self.rest_model(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, ndf, nc, nchannels=1):
        super().__init__()
        self.ndf = ndf
        self.downscale_x = get_downscaling_block(
            nchannels, ndf * 2, 4, 2, 1, use_batch_norm=False
        )
        self.downscale_y = get_downscaling_block(
            nc, ndf * 2, 4, 2, 1, use_batch_norm=False
        )
        self.rest = nn.Sequential(
            get_downscaling_block(
                ndf * 4, ndf * 8, 4, 2, 1
            ),  # input = ndf*2 + ndf*2 concatenation of downscale_x / y
            get_downscaling_block(ndf * 8, ndf * 16, 4, 2, 1),
            get_downscaling_block(ndf * 16, 1, 4, 1, 0, is_last=True),
        )

    def forward(self, x, y):
        y = y.unsqueeze(2).unsqueeze(2).expand(-1, -1, self.ndf, self.ndf)
        y = self.downscale_y(y)
        x = self.downscale_x(x)

        x = torch.cat((x, y), dim=1)
        return self.rest(x).squeeze(1).squeeze(1)  # remove spatial dimensions
