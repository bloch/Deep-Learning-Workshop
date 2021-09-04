import torch.nn as nn
from params import *
from utils.quantization import *
import torch.nn.functional as F

''' Vanilla fully-connected model: one hidden layer '''


class Vanilla(nn.Module):
    def __init__(self, input_size, hidden, model_index):
        super(Vanilla, self).__init__()

        self.path = VANILLA_MODEL_PATH_DIR + "\\block" + str(model_index) + ".pt"
        self.checkpoints_path = VANILLA_MODEL_CHECKPOINTS_PATH_DIR
        self.model_index = model_index
        self.results_dir_path = VANILLA_MODEL_RESULTS_DIR_PATH

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # quantized = quantize1(encoded)
        # dequantized = dequantize1(quantized)
        decoded = self.decoder(encoded)
        return decoded

''' Multilayer fully-connected model: 5 hidden layers '''


class MultiLayer(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, code, model_index):
        super(MultiLayer, self).__init__()

        self.path = MULTILAYER_MODEL_PATH_DIR + "\\block" + str(model_index) + ".pt"
        self.checkpoints_path = MULTILAYER_MODEL_CHECKPOINTS_PATH_DIR
        self.model_index = model_index
        self.results_dir_path = MULTILAYER_MODEL_RESULTS_DIR_PATH

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, code)
        )

        self.decoder = nn.Sequential(
            nn.Linear(code, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # quantized = quantize1(encoded)
        # dequantized = dequantize1(quantized)
        decoded = self.decoder(encoded)
        return decoded


''' convolutional 4-layer model '''


class ConvAutoencoder4(nn.Module):
    def __init__(self):
        super(ConvAutoencoder4, self).__init__()

        self.path = CONVAE4_MODEL_PATH
        self.checkpoints_path = CONVAE4_MODEL_CHECKPOINTS_PATH
        self.results_dir_path = CONVAE4_MODEL_RESULTS_DIR_PATH
        self.path_dir = CONVAE4_MODEL_PATH_DIR

        self.down1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [64,16,200,160]
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # [64,16,100,80]
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [64,32,100,80]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # [64,32,50,40]
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [64,64,50,40]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # [64,64,25,20]
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [64,128,25,20]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # [64,128,12,10]
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # [64,128,25,20]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # [64,64,25,20]
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,64,50,40]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # [64,32,50,40]
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,32,100,80]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  # [64,16,100,80]
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,16,200,160]
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),  # [64,3,200,160]
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )


    def forward(self, x):
        # encoder
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        # decoder
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        return x


''' convolutional 5-layer model '''


class ConvAutoencoder5(nn.Module):
    def __init__(self):
        super(ConvAutoencoder5, self).__init__()

        self.path = CONVAE5_MODEL_PATH
        self.checkpoints_path = CONVAE5_MODEL_CHECKPOINTS_PATH
        self.results_dir_path = CONVAE5_MODEL_RESULTS_DIR_PATH
        self.path_dir = CONVAE5_MODEL_PATH_DIR

        self.down1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [64,16,200,160]
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # [64,16,100,80]
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [64,32,100,80]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # [64,32,50,40]
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [64,64,50,40]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # [64,64,25,20]
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [64,128,25,20]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # [64,128,13,10]
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.down5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [64,256,13,10]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # [64,256,7,5]
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # [64,256,13,10]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # [64,128,13,10]
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # [64,128,25,20]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # [64,64,25,20]
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,64,50,40]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # [64,32,50,40]
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,32,100,80]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  # [64,16,100,80]
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,16,200,160]
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),  # [64,3,200,160]
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

    def forward(self, x):
        # encoder
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        # decoder
        x = self.up5(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        return x

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=16640, z_dim=512):
        super(VAE, self).__init__()

        self.path = VAE_MODEL_PATH
        self.checkpoints_path = VAE_MODEL_CHECKPOINTS_PATH
        self.results_dir_path = VAE_MODEL_RESULTS_DIR_PATH
        self.path_dir = VAE_MODEL_PATH_DIR

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()#,
            #Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            #UnFlatten(),
            # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=2, padding=1),
            # nn.Sigmoid(),
            # nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        flattened_h = torch.reshape(h, (h.size(0), -1))

        z, mu, logvar = self.bottleneck(flattened_h)
        # z is the decoded..
        z = self.fc3(z)
        z = torch.reshape(z, (z.size(0), 128, 13, 10))
        return self.decoder(z), mu, logvar

def loss_fn(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD
