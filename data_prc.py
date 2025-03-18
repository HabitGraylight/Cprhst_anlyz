import python_speech_features as psf
import scipy.io.wavfile as wav

# 加载音频文件
(rate, signal) = wav.read('ZHcn0401.wav')  # 替音频文件路径

# 提取MFCC特征
mfcc_features = psf.mfcc(signal, rate)
print(mfcc_features)