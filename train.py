import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from diffusion import DPMNoiseScheduler
from module import UNetLikeVC, loss_dpm


# ===== Dataset 정의 =====
class VoiceDataset(Dataset):
    def __init__(self, mel_dir, bnf_dir, speaker_id, max_len=512):
        self.mel_paths = sorted(glob.glob(os.path.join(mel_dir, "*.npy")))
        self.bnf_paths = sorted(glob.glob(os.path.join(bnf_dir, "*.npy")))
        self.speaker_id = speaker_id
        self.max_len = max_len
        assert len(self.mel_paths) == len(self.bnf_paths)

    def __len__(self):
        return len(self.mel_paths)

    def __getitem__(self, idx):
        mel = np.load(self.mel_paths[idx])  # (80, T)
        bnf = np.load(self.bnf_paths[idx])  # (T, 144)
        bnf = bnf.T #(144,T)
        T = min(mel.shape[1], bnf.shape[1], self.max_len)

        mel = mel[:, :T]
        bnf = bnf[:, :T]

        return torch.tensor(mel, dtype=torch.float32), \
               torch.tensor(bnf, dtype=torch.float32), \
               self.speaker_id


def collate_fn(batch):
    mel_batch, bnf_batch, speaker_ids = zip(*batch)
    
    # 최대 길이 찾기
    max_len = max(m.shape[1] for m in mel_batch)
    mel_batch = [F.pad(m, (0, max_len - m.shape[1])) for m in mel_batch]  # (80, max_len)

    max_len = max(b.shape[1] for b in bnf_batch)
    bnf_batch = [F.pad(b, (0, max_len - b.shape[1])) for b in bnf_batch]  # (144, max_len)
    

    mel_batch = torch.stack(mel_batch)  # (B, 80, max_len)
    bnf_batch = torch.stack(bnf_batch)  # (B, max_len, 144)
    speaker_ids = torch.tensor(speaker_ids)

    return mel_batch, bnf_batch, speaker_ids


def build_combined_dataset(root_mel_dir, root_bnf_dir, speaker_names):
    datasets = []
    for idx, name in enumerate(speaker_names):
        mel_dir = os.path.join(root_mel_dir, name)
        bnf_dir = os.path.join(root_bnf_dir, name)
        dataset = VoiceDataset(mel_dir, bnf_dir, speaker_id=idx)
        datasets.append(dataset)
    return ConcatDataset(datasets)


# ===== 학습 루프 =====
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습 파라미터
    speaker_list = ['clb', 'bdl', 'slt', 'rms']
    root_mel_dir = "data/mel"
    root_bnf_dir = "data/bnf"
    num_epochs = 30
    batch_size = 16
    num_timesteps = 21
    lr = 1e-3

    # 데이터 로딩
    dataset = build_combined_dataset(root_mel_dir, root_bnf_dir, speaker_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 모델, 스케줄러, 옵티마이저
    model = UNetLikeVC(speakers=len(speaker_list), noise_levels=num_timesteps).to(device)
    scheduler = DPMNoiseScheduler(num_steps=num_timesteps)
    alpha_bar_tensor = scheduler.alphas_bar.to(device)  # shape: (num_timesteps,)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 체크포인트 디렉토리
    os.makedirs("checkpoints", exist_ok=True)

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"[Epoch {epoch+1}]")
        total_loss = 0

        for x_mel, x_bnf, speaker_ids in pbar:
            print(speaker_ids)
            B, _, T = x_mel.shape
            x_mel = x_mel.to(device)
            x_bnf = x_bnf.to(device)
            speaker_ids = speaker_ids.to(device)

            # 무작위 timestep 샘플링
            t = torch.randint(0, num_timesteps, (B,), device=device)

            # DPM loss 계산
            loss = loss_dpm(model, x_0=x_mel, x_bnf=x_bnf, l=t, k=speaker_ids,
                            alpha_bar_lookup=alpha_bar_tensor)
            # Gradient 초기화
            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            # Backpropagation
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}] 평균 손실: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"checkpoints/voicegrad_epoch{epoch+1}.pt")


if __name__ == "__main__":
    train()
