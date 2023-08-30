import torch

import torch.nn as nn

# 定义输入文本

input_text = "Colorful stalls, vibrant fabrics, exotic spices, bustling crowds, haggling vendors, street performers, aromatic food, traditional crafts, lively music, cultural diversity"

#翻译：五颜六色的摊位、充满活力的织物、异国情调的香料、熙熙攘攘的人群、讨价还价的卖家、街头表演者、芳香的食物、传统手工艺品、欢快的音乐、文化多样性。



# 加载预训练的Transformer模型

model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6)

# 将输入文本转换为张量

input_tensor = torch.tensor([input_text])

# 将文本张量输入到Transformer模型中进行计算

output_tensor = model(input_tensor)

# 对Transformer的输出进行均值池化，得到文本特征向量

text_feature_vector = torch.mean(output_tensor, dim=1)

# 打印文本特征向量的形状

print(text_feature_vector.shape)