import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载音频（统一采样率）
file_path = 'delicious.mp3'
y, sr = librosa.load(file_path, sr=22050)  #强制重采样为22050Hz

# 优化MFCC参数
n_fft = 256                 # 2的幂次
hop_length = 128            # 帧移=FFT长度/2
n_mels = 40                 # 梅尔滤波器数量
n_mfcc = 13                 # 保留的MFCC系数

# 提取MFCC（显式指定梅尔滤波器）
mfccs = librosa.feature.mfcc(
    y=y, sr=sr,
    n_mfcc=n_mfcc,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels           #控制梅尔滤波器数量
)

# 去除能量系数（第0阶）
mfccs = mfccs[1:, :]        # 通常丢弃第0阶系数

# 标准化
scaler = StandardScaler()
mfccs_scaled = scaler.fit_transform(mfccs.T).T  # 时间轴标准化

# 可视化（修正时间轴）
plt.figure(figsize=(12, 6))
librosa.display.specshow(
    mfccs_scaled,
    x_axis='time',
    y_axis='mel', # 以梅尔频率刻度
    sr=sr,                  # 传递采样率
    hop_length=hop_length,   # 传递帧移
    cmap='viridis',         # 优化颜色映射
    vmin=-3,                # 动态范围限制
    vmax=3
)
plt.colorbar(format='%+2.0f dB')
plt.title(f'MFCC (n_mfcc={n_mfcc}, n_mels={n_mels})')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
librosa.display.specshow(mel_basis, sr=sr, hop_length=n_fft, x_axis='linear')
plt.colorbar()
plt.title('Mel Filters')
plt.show()