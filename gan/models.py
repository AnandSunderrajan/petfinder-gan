import torch
from gan.spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        #convolutional layer with in_channels=3, out_channels=128, kernel=4, stride=2
        self.conv1 = SpectralNorm(torch.nn.Conv2d(3, 128, 4, 2, padding=1))
        #convolutional layer with in_channels=128, out_channels=256, kernel=4, stride=2
        self.conv2 = SpectralNorm(torch.nn.Conv2d(128, 256, 4, 2, padding=1))
        #batch norm
        self.batch1 = torch.nn.BatchNorm2d(256)
        #convolutional layer with in_channels=256, out_channels=512, kernel=4, stride=2
        self.conv3 = SpectralNorm(torch.nn.Conv2d(256, 512, 4, 2, padding=1))
        #batch norm
        self.batch2 = torch.nn.BatchNorm2d(512)
        #convolutional layer with in_channels=512, out_channels=1024, kernel=4, stride=2
        self.conv4 = SpectralNorm(torch.nn.Conv2d(512, 1024, 4, 2, padding=1))
        #batch norm
        self.batch3 = torch.nn.BatchNorm2d(1024)
        #convolutional layer with in_channels=1024, out_channels=1, kernel=4, stride=1
        self.conv5 = SpectralNorm(torch.nn.Conv2d(1024, 1, 4, 2))

        # #convolutional layer with in_channels=3, out_channels=128, kernel=4, stride=2
        # self.conv1 = torch.nn.Conv2d(3, 128, 4, 2, padding=1)
        # #convolutional layer with in_channels=128, out_channels=256, kernel=4, stride=2
        # self.conv2 = torch.nn.Conv2d(128, 256, 4, 2, padding=1)
        # #batch norm
        # self.batch1 = torch.nn.BatchNorm2d(256)
        # #convolutional layer with in_channels=256, out_channels=512, kernel=4, stride=2
        # self.conv3 = torch.nn.Conv2d(256, 512, 4, 2, padding=1)
        # #batch norm
        # self.batch2 = torch.nn.BatchNorm2d(512)
        # #convolutional layer with in_channels=512, out_channels=1024, kernel=4, stride=2
        # self.conv4 = torch.nn.Conv2d(512, 1024, 4, 2, padding=1)
        # #batch norm
        # self.batch3 = torch.nn.BatchNorm2d(1024)
        # #convolutional layer with in_channels=1024, out_channels=1, kernel=4, stride=1
        # self.conv5 = torch.nn.Conv2d(1024, 1, 4, 2)

    def forward(self, x):
        # print(x.size())
        x = torch.nn.LeakyReLU(0.2)(self.conv1(x))
        # print(x.size())
        x = torch.nn.LeakyReLU(0.2)(self.conv2(x))
        # print(x.size())
        x = self.batch1(x)
        # print(x.size())
        x = torch.nn.LeakyReLU(0.2)(self.conv3(x))
        # print(x.size())
        x = self.batch2(x)
        # print(x.size())
        x = torch.nn.LeakyReLU(0.2)(self.conv4(x))
        # print(x.size())
        x = self.batch3(x)
        # print(x.size())
        x = torch.nn.LeakyReLU(0.2)(self.conv5(x))
        # print(x.size())
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        self.model = torch.nn.Sequential(
        # transpose convolution with in_channels=NOISE_DIM, out_channels=1024, kernel=4, stride=1
            torch.nn.ConvTranspose2d(noise_dim, 1024, 4, 1),
            torch.nn.ReLU(),
        # batch norm
            torch.nn.BatchNorm2d(1024),
        # transpose convolution with in_channels=1024, out_channels=512, kernel=4, stride=2
            torch.nn.ConvTranspose2d(1024, 512, 4, 2, padding=1),
            torch.nn.ReLU(),
        # batch norm
            torch.nn.BatchNorm2d(512),
        # transpose convolution with in_channels=512, out_channels=256, kernel=4, stride=2
            torch.nn.ConvTranspose2d(512, 256, 4, 2, padding=1),
            torch.nn.ReLU(),
        # batch norm
            torch.nn.BatchNorm2d(256),
        # transpose convolution with in_channels=256, out_channels=128, kernel=4, stride=2
            torch.nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            torch.nn.ReLU(),
        # batch norm
            torch.nn.BatchNorm2d(128),
        # transpose convolution with in_channels=128, out_channels=3, kernel=4, stride=2
            torch.nn.ConvTranspose2d(128, 3, 4, 2, padding=1),
            torch.nn.Tanh())



    def forward(self, x):
        return self.model(x.view(-1, self.noise_dim, 1, 1))
    

