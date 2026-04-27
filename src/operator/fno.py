import torch
import torch.nn as nn
import numpy as np


class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT → linear transform → inverse FFT."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x)

        out_fft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        modes_h = min(self.modes1, H)
        modes_w = min(self.modes2, W // 2 + 1)
        out_fft[:, :, :modes_h, :modes_w] = self.compl_mul2d(
            x_fft[:, :, :modes_h, :modes_w], self.weights1[:, :, :modes_h, :modes_w]
        )
        out_fft[:, :, -modes_h:, :modes_w] = self.compl_mul2d(
            x_fft[:, :, -modes_h:, :modes_w], self.weights2[:, :, :modes_h, :modes_w]
        )

        return torch.fft.irfft2(out_fft, s=(H, W))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: condition on parameter vector."""

    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, num_features * 2),
        )

    def forward(self, x, cond):
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class FNOBlock(nn.Module):
    """FNO block: SpectralConv + FiLM conditioning + residual connection."""

    def __init__(self, in_channels, out_channels, modes1, modes2, cond_dim):
        super().__init__()
        self.spec = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.film = FiLMLayer(out_channels, cond_dim)
        self.act = nn.GELU()

    def forward(self, x, cond):
        x1 = self.spec(x)
        x2 = self.conv(x)
        out = x1 + x2
        out = self.film(out, cond)
        return self.act(out)


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2D field prediction.

    Input: (params, c_rate) → encoded conditioning vector
    Output: predicted fields on (x, t) grid

    For voltage prediction: uses a separate head that pools over spatial dims.
    """

    def __init__(
        self,
        num_params=7,
        in_channels=2,
        out_channels=3,
        mid_channels=64,
        num_layers=4,
        modes1=16,
        modes2=32,
        cond_dim=128,
    ):
        super().__init__()
        self.num_params = num_params
        self.cond_dim = cond_dim

        self.param_encoder = nn.Sequential(
            nn.Linear(num_params + 1, 64),
            nn.GELU(),
            nn.Linear(64, cond_dim),
            nn.GELU(),
        )

        self.lifting = nn.Conv2d(in_channels, mid_channels, 1)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(FNOBlock(mid_channels, mid_channels, modes1, modes2, cond_dim))

        self.projection = nn.Sequential(
            nn.Conv2d(mid_channels, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1),
        )

        self.voltage_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Flatten(2),
            nn.Conv1d(mid_channels, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 1, 1),
        )

    def forward(self, field_input, params, c_rate):
        """
        Args:
            field_input: (B, C_in, H, W) - coordinate grid + initial conditions
            params: (B, N_params) - parameter vector
            c_rate: (B, 1) - C-rate
        Returns:
            fields: (B, C_out, H, W) - predicted fields
            voltage: (B, Nt) - terminal voltage prediction
        """
        cond = torch.cat([params, c_rate], dim=-1)
        cond = self.param_encoder(cond)

        x = self.lifting(field_input)

        for block in self.blocks:
            x = x + block(x, cond)

        fields = self.projection(x)

        B, C, H, W = x.shape
        v_pred = self.voltage_head(x).squeeze(1)

        return fields, v_pred
