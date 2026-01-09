import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size, stride, padding)

    def forward(self, x):
        x_proj = self.conv(x)
        out, gate = x_proj.chunk(2, dim=1)
        return out * torch.sigmoid(gate)


class DeconvGLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = (kernel_size - stride) // 2
        output_padding = stride - 1
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels * 2, kernel_size, stride,
                                         padding=padding, output_padding=output_padding)

    def forward(self, x):
        x_proj = self.deconv(x)
        out, gate = x_proj.chunk(2, dim=1)
        return out * torch.sigmoid(gate)


class UNetLikeVC(nn.Module):
    def __init__(self,
                 mel_dim=80,
                 bnf_dim=144,
                 noise_levels=21,
                 speakers=4,
                 cond_dim=16,
                 bnf_out_dim=32,
                 base_channels=512):
        super().__init__()
        mel_mean = np.load("mel_mean.npy").astype(np.float32)  # (144,)
        mel_std = np.load("mel_std.npy").astype(np.float32)    # (144,)
        self.register_buffer("mel_mean", torch.from_numpy(mel_mean).view(1, -1, 1))  # (1, 144, 1)
        self.register_buffer("mel_std", torch.from_numpy(mel_std).view(1, -1, 1))

        self.noise_embed = nn.Embedding(noise_levels, cond_dim)
        self.speaker_embed = nn.Embedding(speakers, cond_dim)
        self.bnf_proj = nn.Conv1d(bnf_dim, bnf_out_dim, kernel_size=3, stride=1, padding=1)

        cond_total_dim = cond_dim * 2 + bnf_out_dim
        in_channels = mel_dim + cond_total_dim

        # Encoder layers
        self.encoder = nn.ModuleList([
            GLUBlock(in_channels, base_channels, kernel_size=9, stride=1),
            GLUBlock(base_channels + cond_total_dim, base_channels, kernel_size=8, stride=2),
            GLUBlock(base_channels + cond_total_dim, base_channels, kernel_size=9, stride=1),
            GLUBlock(base_channels + cond_total_dim, base_channels, kernel_size=8, stride=2),
            GLUBlock(base_channels + cond_total_dim, base_channels, kernel_size=5, stride=1),
            GLUBlock(base_channels + cond_total_dim, base_channels, kernel_size=5, stride=1),
        ])
        self.bottleneck = GLUBlock(base_channels + cond_total_dim, base_channels, kernel_size=5, stride=1)
        # Decoder layers
        self.decoder = nn.ModuleList([
            GLUBlock(base_channels * 2 + cond_total_dim, base_channels, kernel_size=5, stride=1),
            DeconvGLUBlock(base_channels * 2 + cond_total_dim, base_channels, kernel_size=8, stride=2),
            GLUBlock(base_channels * 2 + cond_total_dim, base_channels, kernel_size=9, stride=1),
            DeconvGLUBlock(base_channels * 2 + cond_total_dim, base_channels, kernel_size=8, stride=2),
            nn.Conv1d(base_channels *2 + cond_total_dim, mel_dim, kernel_size=9, stride=1, padding=4)
        ])

    def forward(self, x_mel, x_bnf, l, k):
        B, _, T = x_mel.size()
        x_mel = (x_mel - self.mel_mean) / self.mel_std 

        # Conditioning
        e_l = self.noise_embed(l).unsqueeze(-1).repeat(1, 1, T)
        e_k = self.speaker_embed(k).unsqueeze(-1).repeat(1, 1, T)
        bnf_cond = self.bnf_proj(x_bnf)

        x = x_mel
        

        # Encoder
        skips = []
        count = 0 
        for enc in self.encoder:

            e_l_r = F.interpolate(e_l, size=x.size(-1), mode='nearest')
            e_k_r = F.interpolate(e_k, size=x.size(-1), mode='nearest')
            bnf_r = F.interpolate(bnf_cond, size=x.size(-1), mode='nearest')
            x = torch.cat([x, e_l_r, e_k_r, bnf_r], dim=1)
            x = enc(x)
            if count < 5:
                skips.append(x)
            count = count+1

        e_l_r = F.interpolate(e_l, size=x.size(-1), mode='nearest')
        e_k_r = F.interpolate(e_k, size=x.size(-1), mode='nearest')
        bnf_r = F.interpolate(bnf_cond, size=x.size(-1), mode='nearest')
        x = torch.cat([x, e_l_r, e_k_r, bnf_r], dim=1)
        x = self.bottleneck(x)

        # Decoder
        count = 0
        for dec, skip in zip(self.decoder, reversed(skips)):
            if x.size(-1) != skip.size(-1):
                x = F.pad(x, (0, skip.size(-1) - x.size(-1)))

            e_l_d = F.interpolate(e_l, size=x.size(-1), mode='nearest')
            e_k_d = F.interpolate(e_k, size=x.size(-1), mode='nearest')
            bnf_d = F.interpolate(bnf_cond, size=x.size(-1), mode='nearest')


            x = torch.cat([x, skip, e_l_d, e_k_d, bnf_d], dim=1)
            x = dec(x)
            count = count + 1
        

        return x

def loss_dpm(model, x_0, x_bnf, l, k, alpha_bar_lookup):
    """
    model: UNetLikeVC
    x_0: (B, mel_dim, T) - 원본 mel-spectrogram
    l: (B,) - noise level index
    k: (B,) - speaker index
    alpha_bar_lookup: dict or tensor mapping l → ᾱ_l
    """
    print("loss_dpm")
    B, C, T = x_0.shape

    # 노이즈 샘플링
    epsilon = torch.randn_like(x_0)

    # alpha_bar 값 가져오기 (예: precomputed tensor)
    alpha_bar_l = alpha_bar_lookup[l].view(B, 1, 1)  # (B, 1, 1)

    # noisy input 생성 (forward diffusion sample)
    noisy_x = (alpha_bar_l.sqrt() * x_0 +
               (1.0 - alpha_bar_l).sqrt() * epsilon)


    # 모델 예측 (노이즈 추정)
    pred_epsilon = model(x_mel=noisy_x, x_bnf=x_bnf, l=l, k=k)  # bnf는 따로 처리 필요

    # L1 loss 계산
    loss = F.l1_loss(pred_epsilon, epsilon)

    return loss

# 아래 코드는 DPM 스케줄러 관련 구현 코드인데, 다른 파일에서 구현해 놓아서 주석처리 해놨습니다.   
# def f(l, L):
#     return np.cos(((l / 20) + 0.008) / (1 + 0.008) * (np.pi / 2))**2

# def alpha_bar(l):
#     return f(l)/f(0)

# def beta(l):
#     beta_l = 1-alpha_bar(l)/alpha_bar(l-1)
#     if beta_l > 0.999:
#         beta_l = 0.999
#     return beta_l

# def v(l):
#     return math.sqrt(beta(l))