import statsmodels.api as sm

# 构建公式字符串：EmotionScore 由 MFCC1 和 MFCC2 预测
fixed_formula = "EmotionScore ~ MFCC1 + MFCC2"

# 使用 variance components (vc_formula) 指定多重随机效应
vc = {
    "Speaker": "0 + C(Speaker)",   # 说话人随机效应
    "Language": "0 + C(Language)"  # 语言随机效应
}

# 创建线性混合效应模型
# groups=np.ones(len(df)) 相当于将所有数据视为一个总体组，
# 我们将具体的随机效应结构通过 vc_formula 指定
model = sm.MixedLM.from_formula(fixed_formula, data=df, groups=np.ones(df.shape[0]), vc_formula=vc)
result = model.fit(reml=True)  # 使用REML估计
print(result.summary())