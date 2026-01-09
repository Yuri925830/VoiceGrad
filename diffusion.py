import torch
import torch.nn as nn
import numpy as np

class DPMNoiseScheduler:
    def __init__(self, num_steps=20, eta=0.008):
        """
        num_steps: total diffusion step number
        eta: small offset, for cosine schedule stability. 
        """
        self.num_steps = num_steps
        self.eta = eta
        self.register_schedule()

    def register_schedule(self):
        """
        cosine-based beta schedule from the paper
        """
        steps = self.num_steps
        eta = self.eta
        timesteps = torch.arange(0, steps + 1, dtype=torch.float32)

        f = lambda t: torch.cos((((t / steps) + eta)/(1 + eta)) * (np.pi / 2)) ** 2
        alphas_bar = f(timesteps) / f(torch.tensor(0.0))

        self.alphas_bar = alphas_bar  # shape: (steps + 1,)
        self.alphas = alphas_bar[1:] / alphas_bar[:-1]
        self.betas = torch.clamp(1.0 - self.alphas, max = 0.999)
        self.alphas_bar = alphas_bar[1:]  # remove alpha_bar[0], match index with step

        # For noise generation
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

    def forward(self, x0: torch.Tensor, t: torch.LongTensor):
        """
        Args:
            x0: (B, D, T), original clean mel-spectrogram
            t: (B,), timestep indices (0 ~ num_steps-1)
        Returns:
            x_t: (B, D, T), noisy version
            noise: (B, D, T), added noise ε
        """
        device = x0.device
        B, D, T = x0.shape
        noise = torch.randn_like(x0)

        # gather alpha_bar values per t
        sqrt_ab = self.sqrt_alphas_bar.to(device)[t].view(B, 1, 1)
        sqrt_1m_ab = self.sqrt_one_minus_alphas_bar.to(device)[t].view(B, 1, 1)

        x_t = sqrt_ab * x0 + sqrt_1m_ab * noise
        return x_t, noise

if __name__ == "__main__":

    # 클래스 정의는 생략하고 이미 위에 있다고 가정
    # from diffusion import DPMNoiseScheduler

    # 1. .npy 파일 불러오기
    input_path = 'data/mel/bdl/arctic_a0001.npy'   # 테스트용 경로, 원하는 대로 수정하시면 됩니다.
    output_path = 'test.npy'  # 저장할 파일 경로

    # (B, D, T) 형태로 로드
    x0_np = np.load(input_path)  # 예: (1, 80, 200)
    x0_tensor = torch.tensor(x0_np, dtype=torch.float32)
    x0_tensor = x0_tensor.unsqueeze(0)
    # 2. DPMNoiseScheduler 초기화 및 forward 수행
    scheduler = DPMNoiseScheduler(num_steps=20)
    t = torch.full((x0_tensor.shape[0],), 11, dtype=torch.long)  # 배치마다 t=11

    # 3. 노이즈 추가
    x_t, noise = scheduler.forward(x0_tensor, t)

    # 4. 결과 저장
    x_t_np = x_t.squeeze(0).numpy()
    np.save(output_path, x_t_np)
    print(f"Noised mel-spectrogram saved at: {output_path}")
