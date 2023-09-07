import torch 
import torch.nn as nn
import numpy as np

class ResidualConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, is_res: bool=False) -> None:

        super().__init__()

        self.same_channels = in_channels == out_channels

        self.is_res = is_res

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.conv1(x)
        out = self.conv2(x1)

        if self.is_res:
            if self.same_channels:
                out += x
            else:
                shortcut = nn.Conv2d(x.shape[1], out.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out += shortcut(x)
            out /= 1.414

        return out
    
class UNetUp(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:

        super().__init__()
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:

        x = torch.cat((x, skip), 1)
        return self.model(x)

class UNetDown(nn.Module):

    def __init__(self, in_channels:int, out_channels: int) -> None:

        super().__init__()

        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
            return self.model(x)

class EmbedFC(nn.Module):
    
    def __init__(self, input_dim, emb_dim):

        super().__init__()

        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim), 
            nn.GELU(), 
            nn.Linear(emb_dim, emb_dim)
            )
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUNet(nn.Module):

    def __init__(self, in_channels: int, n_feat: int=256, n_cfeat: int=10, height: int=28):

        super().__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.height = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UNetDown(n_feat, n_feat)
        self.down2 = UNetDown(n_feat, 2*n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.time_embed1 = EmbedFC(1, 2*n_feat)
        self.time_embed2 = EmbedFC(1, 1*n_feat)
        self.contex_embed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contex_embed2 = EmbedFC(n_cfeat, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*n_feat, 2*n_feat, self.height//4, self.height//4),
            nn.GroupNorm(8, 2*n_feat),
            nn.ReLU()
        )

        self.up1 = UNetUp(4*n_feat, n_feat)
        self.up2 = UNetUp(2*n_feat, n_feat)

        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor=None):

        x = self.init_conv(x)  # [B, 256, 28, 28]
        down1 = self.down1(x)  # [B, 256, 14, 14]
        down2 = self.down2(down1)  # [B, 512, 7, 7]

        hidden_vec = self.to_vec(down2)  # [B, 512, 1, 1]

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x.device)

        cembed1 = self.contex_embed1(c).view(-1, 2*self.n_feat, 1, 1)  # [B, 512]
        tembed1 = self.time_embed1(t).view(-1, 2*self.n_feat, 1, 1)  # [B, 512]
        cembed2 = self.contex_embed2(c).view(-1, self.n_feat, 1, 1)  # [B, 256]
        tembed2 = self.time_embed2(t).view(-1, self.n_feat, 1, 1)  # [B, 256]

        up1 = self.up0(hidden_vec)  # [B, 512, 7, 7]
        up2 = self.up1(cembed1 * up1 + tembed1, down2)  # [B, 256, 14, 14]
        up3 = self.up2(cembed2 * up2 + tembed2, down1)  # [B, 256, 28, 28]
        out = self.out_conv(torch.cat((up3, x), dim=1))  # [B, 3, 28, 28]

        return out

class DDPM(nn.Module):

    def __init__(self, in_channels, n_feat, nc_feat, height, beta_1, beta_2, timesteps, save_dir, device='cpu'):

        super().__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.nc_feat = nc_feat
        self.height = height
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.timesteps = timesteps
        self.save_dir = save_dir
        self.device = device

        self.context_unet = ContextUNet(in_channels, n_feat, nc_feat, height).to(device)
        self.b_t = (beta_2 - beta_1) * torch.linspace(0, 1, timesteps + 1).to(self.device) + beta_1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()
        self.ab_t[0] = 1

    def predict_noise(self, x, t, c=None):
        return self.context_unet(x, t, c)

    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t[t].sqrt() * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise
    
    def sample_ddpm(self, n_sample, contexts=None, save_rate=20):

        samples = torch.randn(n_sample, self.in_channels, self.height, self.height).to(self.device)
        intermediate = []

        for i in range(self.timesteps, 0, -1):

            print("Sampling timestep: ", i, "/", self.timesteps, end="\r")

            t = torch.tensor([i / self.timesteps])[:, None, None, None].to(self.device)

            z = torch.randn_like(samples) if i > 1 else 0

            eps = self.predict_noise(samples, t, contexts)
            samples = self.denoise_add_noise(samples, i, eps, z)

            if i % save_rate == 0 or i==self.timesteps:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)

        return samples, intermediate

    def perturb_input(self, x: torch.Tensor, t:torch.Tensor, noise: float):
        """
        Return a perturbed input.
        """
        return self.ab_t.sqrt()[t, None, None, None] * x + (1 - self.ab_t[t, None, None, None]) * noise