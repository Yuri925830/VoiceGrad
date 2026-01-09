import os
import torch
import numpy as np
from module import UNetLikeVC
from diffusion import DPMNoiseScheduler
import soundfile as sf

@torch.no_grad()
def sample_dpm(model, scheduler, x_start, bnf, speaker_id, num_steps=21, start_step=10):
    """
    x_start: (1, 80, T) - 초기 mel-spectrogram (source speaker)
    bnf: (1, 144, T) - BNF conditioning from same source
    speaker_id: int - target speaker index
    start_step: int - L′ (diffusion 중간에서 시작)
    """
    device = x_start.device
    x = x_start.clone()

    alphas = scheduler.alphas.to(device)
    alpha_bars = scheduler.alphas_bar.to(device)
    betas = scheduler.betas.to(device)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars).to(device)

    for l in reversed(range(start_step)):
        l_tensor = torch.full((1,), l, device=device, dtype=torch.long)
        k_tensor = torch.full((1,), speaker_id, device=device, dtype=torch.long)
        z = torch.randn_like(x) if l > 0 else torch.zeros_like(x)

        eps_theta = model(x, bnf, l_tensor, k_tensor)

        coef = (1 - alphas[l]) / sqrt_one_minus_alpha_bar[l]
        x = (1.0 / torch.sqrt(alphas[l])) * (x - coef * eps_theta) + torch.sqrt(betas[l]) * z

    return x  # converted mel-spectrogram


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== 모델 로드 ======
    model = UNetLikeVC(speakers=4, noise_levels=21).to(device)
    model.load_state_dict(torch.load("checkpoints/voicegrad_epoch50.pt")) # 체크포인트 수정 요망
    model.eval()

    scheduler = DPMNoiseScheduler(num_steps=21)

    # ====== 입력 데이터 ======
    mel_path = "samples/source_mel.npy"
    bnf_path = "samples/source_bnf.npy"
    target_speaker = 2  # 예: slt

    mel = torch.tensor(np.load(mel_path), dtype=torch.float32).unsqueeze(0).to(device)  # (1, 80, T)
    bnf = torch.tensor(np.load(bnf_path), dtype=torch.float32).unsqueeze(0).to(device)  # (1, 144, T)

    # ====== 변환 수행 ======
    converted_mel = sample_dpm(model, scheduler, mel, bnf, speaker_id=target_speaker)

    # ====== 결과 저장 ======
    converted_mel_np = converted_mel.squeeze(0).cpu().numpy()
    np.save("samples/converted_mel.npy", converted_mel_np)
    print("✅ Converted mel-spectrogram saved to samples/converted_mel.npy")

    # (선택) HiFi-GAN으로 waveform 변환 후 저장
    # from hifi_gan.vocoder import HiFiGANVocoder
    # vocoder = HiFiGANVocoder().to(device)
    # wav = vocoder(converted_mel)  # (1, samples)
    # sf.write("samples/converted.wav", wav.squeeze().cpu().numpy(), samplerate=16000)


if __name__ == "__main__":
    main()
