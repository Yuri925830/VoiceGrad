import torch
import numpy as np
import argparse
from diffusion import DPMNoiseScheduler
from module import UNetLikeVC

@torch.no_grad()
def reverse_diffusion(model, scheduler, x, x_bnf, k, start_step=11):
    for l in reversed(range(start_step + 1)):
        B, D, T = x.shape
        t_tensor = torch.full((B,), l, device=x.device, dtype=torch.long)

        eps = model(x, x_bnf, t_tensor, k)  # (B, D, T)

        alpha_l = scheduler.alphas.to(x.device)[l].view(B, 1, 1)
        alpha_bar_l = scheduler.alphas_bar.to(x.device)[l].view(B, 1, 1)
        beta_l = scheduler.betas.to(x.device)[l].view(B, 1, 1)
        v_l = torch.sqrt(beta_l)

        term1 = (1 - alpha_l) / torch.sqrt(1 - alpha_bar_l) * eps
        x = (1 / torch.sqrt(alpha_l)) * (x - term1)

        if l > 0:
            z = torch.randn_like(x)
            x += v_l * z
    return x

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 원본 x_0 및 BNF 불러오기
    x_0 = torch.tensor(np.load(args.x0), dtype=torch.float32).unsqueeze(0).to(device)      # (1, 80, T)
    x_bnf = torch.tensor(np.load(args.bnf), dtype=torch.float32).T.unsqueeze(0).to(device)  # (1, 144, T)
    speaker = torch.tensor([args.spk_id], dtype=torch.long).to(device)

    # 2. 모델 및 스케줄러 로딩
    model = UNetLikeVC(speakers=args.num_speakers, noise_levels=args.num_timesteps).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    scheduler = DPMNoiseScheduler(num_steps=args.num_timesteps)

    # 3. x_0 → x_t로 노이즈 추가
    t = torch.full((x_0.shape[0],), args.start_step, dtype=torch.long, device=device)  # (B,)
    x_t, _ = scheduler.forward(x_0, t)  # 노이즈 추가된 x_t

    # 4. reverse diffusion 수행
    x_0_hat = reverse_diffusion(model, scheduler, x_t, x_bnf, speaker, start_step=args.start_step)

    # 5. 결과 저장
    np.save(args.output_npy, x_0_hat.squeeze(0).cpu().numpy())
    print(f"✅ 변환된 mel-spectrogram 저장 완료: {args.output_npy}")


# 테스트할 때마다 수정했던 코드라 default 값에 특정 파일 명 있는 거는 원하시는대로 수정해주시면 됩니다!
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x0", default = "data/mel/clb/arctic_a0001.npy", help="원본 mel-spectrogram (npy, shape=(80, T))")
    parser.add_argument("--bnf", default = "data/bnf/clb/arctic_a0001.ling_feat.npy", help="BNF (npy, shape=(T, 144))")
    parser.add_argument("--spk_id", type=int, default = 3, help="타겟 화자 ID")
    parser.add_argument("--ckpt", default = "checkpoints/voicegrad_epoch50.pt", help="모델 checkpoint 경로")
    parser.add_argument("--output_npy", default="converted.npy")
    parser.add_argument("--start_step", type=int, default=11)
    parser.add_argument("--num_timesteps", type=int, default=21)
    parser.add_argument("--num_speakers", type=int, default=4)
    args = parser.parse_args()
    main(args)
