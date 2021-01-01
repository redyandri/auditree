# import torch
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())

import re
import textract
#read the content of pdf as text
fp=r"../../corpus/bpkhackathon/telkom/FS Q12018_INDONESIA.pdf"
text = textract.process(fp)
#use four space as paragraph delimiter to convert the text into list of paragraphs.
print (re.split('\s{4,}',text))