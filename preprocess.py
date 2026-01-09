import os
import librosa
import numpy as np
from tqdm import tqdm

def wav_to_logmel(wav_path, sr=16000, n_mels=80, win_size=1024, hop_size=256):
    y, _ = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_size,
                                         hop_length=hop_size, n_mels=n_mels, center=False)
    logmel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return logmel  # shape: (80, T)

def process_directory(wav_root, save_root, speaker_list):
    for speaker in speaker_list:
        wav_dir = os.path.join(wav_root, speaker,"wav")
        save_dir = os.path.join(save_root, speaker)
        os.makedirs(save_dir, exist_ok=True)

        wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

        print(f"[{speaker}] {len(wav_files)} files to process.")

        for wav_file in tqdm(wav_files, desc=f"Extracting {speaker}"):
            wav_path = os.path.join(wav_dir, wav_file)
            logmel = wav_to_logmel(wav_path)
            save_path = os.path.join(save_dir, wav_file.replace(".wav", ".npy"))
            np.save(save_path, logmel)

    print("✅ All log-mel features extracted and saved.")


if __name__ == "__main__":
    # 예: data/raw/clb/*.wav → data/mel/clb/*.npy
    wav_root = "data/wav"     # 입력 wav 경로
    save_root = "data/mel"    # 저장할 log-mel 경로
    speaker_list = ["clb", "bdl", "slt", "rms", "jmk", "ksp", "lnh"]  # speaker 폴더 이름

    process_directory(wav_root, save_root, speaker_list)
