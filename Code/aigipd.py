import torch

import torch.nn as nn

# ���������ı�

input_text = "Colorful stalls, vibrant fabrics, exotic spices, bustling crowds, haggling vendors, street performers, aromatic food, traditional crafts, lively music, cultural diversity"

#���룺������ɫ��̯λ������������֯������������ϡ�������������Ⱥ���ּۻ��۵����ҡ���ͷ�����ߡ������ʳ���ͳ�ֹ���Ʒ����������֡��Ļ������ԡ�



# ����Ԥѵ����Transformerģ��

model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6)

# �������ı�ת��Ϊ����

input_tensor = torch.tensor([input_text])

# ���ı��������뵽Transformerģ���н��м���

output_tensor = model(input_tensor)

# ��Transformer��������о�ֵ�ػ����õ��ı���������

text_feature_vector = torch.mean(output_tensor, dim=1)

# ��ӡ�ı�������������״

print(text_feature_vector.shape)