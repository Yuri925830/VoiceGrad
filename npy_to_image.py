# This is the code for visualizing mel-spectrogram

import numpy as np
import matplotlib.pyplot as plt
import os

def npy_to_image(npy_path, save_path, cmap='viridis'):
    logmel = np.load(npy_path)

    # 시각화를 위한 설정
    plt.figure(figsize=(10, 4))
    plt.imshow(logmel, origin='lower', aspect='auto', cmap=cmap)
    plt.axis('off')  # 축 제거
    plt.tight_layout(pad=0)
    
    # 이미지 저장
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 예시
if __name__ == "__main__" :
    npy_path = "data/bnf/arctic_a0001.ling_feat.npy"
    save_path = "data/mel_img/dlb/example.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    npy_to_image(npy_path, save_path)
