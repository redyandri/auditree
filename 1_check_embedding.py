from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from scipy import spatial

MAX_TOKEN_LENTH=59
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
a="aku main sepak bola"
b="aku main bulu tangkis"
c="aku tidak suka olah raga"
tokena=tokenizer.encode(a,add_special_tokens=True)
tokenb=tokenizer.encode(b,add_special_tokens=True)
tokenc=tokenizer.encode(c,add_special_tokens=True)

tokens=[tokena,tokenb,tokenc]
max_len = 0
for i in tokens:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokens])

print(padded)
attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()

# for f in features:
#     print(f)

distab = 1 - spatial.distance.cosine(features[0], features[1])
distac = 1 - spatial.distance.cosine(features[0], features[2])
distbc = 1 - spatial.distance.cosine(features[1], features[2])
print(distab)
print(distac)
print(distbc)