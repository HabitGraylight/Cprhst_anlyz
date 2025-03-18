import numpy as np
import pandas as pd

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 定义类别
emotions = ['negative', 'neutral', 'positive']
languages = ['English', 'Chinese', 'Spanish']  # 假设3种语言
n_speakers = 30   # 假设30个说话人
# 每个说话人随机分配一种语言（每个说话人只说一种语言）
speaker_languages = {f"Speaker_{i}": np.random.choice(languages) for i in range(1, n_speakers+1)}

# 模拟每条语音记录
n_records = 500  # 总记录数
data = []
for i in range(n_records):
    # 随机选择说话人和对应语言
    speaker = np.random.choice(list(speaker_languages.keys()))
    language = speaker_languages[speaker]
    # 随机选择情感类别
    emotion_label = np.random.choice(emotions, p=[0.3, 0.4, 0.3])  # 赋予不同概率以模拟分布
    # 将情感类别映射为情感极性分数
    if emotion_label == 'negative':
        emotion_score = -1
    elif emotion_label == 'positive':
        emotion_score = 1
    else:
        emotion_score = 0
    # 模拟 MFCC 特征: 不同情感下特征均值略有差异
    # 假设我们使用 MFCC1 和 MFCC2 两个特征进行演示
    if emotion_label == 'positive':
        mfcc1 = np.random.normal(loc=12.0, scale=2.0)  # 积极情绪可能平均音调更高
        mfcc2 = np.random.normal(loc=5.0, scale=1.0)   # 积极情绪可能能量更高
    elif emotion_label == 'negative':
        mfcc1 = np.random.normal(loc=8.0, scale=2.0)   # 消极情绪音调偏低
        mfcc2 = np.random.normal(loc=3.0, scale=1.0)   # 消极情绪能量偏低
    else:  # neutral
        mfcc1 = np.random.normal(loc=10.0, scale=2.0)  # 中性情绪居中
        mfcc2 = np.random.normal(loc=4.0, scale=1.0)
    # 加入说话人和语言的随机影响（比如说话人习惯、语言特性造成MFCC整体偏移）
    # 这里通过在MFCC上增加小的随机偏差来模拟
    speaker_offset = np.random.normal(scale=0.5)   # 说话人风格对MFCC的影响
    language_offset = np.random.normal(scale=0.5)  # 语言环境对MFCC的影响
    mfcc1 += speaker_offset + language_offset
    mfcc2 += speaker_offset + language_offset
    data.append([speaker, language, emotion_label, emotion_score, mfcc1, mfcc2])

# 创建DataFrame
df = pd.DataFrame(data, columns=['Speaker', 'Language', 'EmotionLabel', 'EmotionScore', 'MFCC1', 'MFCC2'])
# 查看前5行数据
print(df.head())