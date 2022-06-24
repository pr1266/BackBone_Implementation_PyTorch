import torch
import torch.nn as nn

"""
miresim be architecture mobile net
idea asli paper moarefi "DepthWise Seperable Conv Layer" ast
in dw layer az 2 bakhsh tashkil shode:
1 - marhale aval miaim be jaye conv az depthwise conv estefade mikonim
yani har kernel faghat ba yeki az filter ha ke filter e naziresh hast conv mishe
yani age voroodi 28 * 28 * 196 e, miaim 196 ta kernel estefade mikonim va
har kodoom az filter ha ba ye kernel conv mishe.
2 - hala baadesh miaim az conv 1*1 estefade mikonim ta feature map haye
dar dastres ro summarize konim va hajm mohasebat biad paiin ta beshe azash
too app haye mobile estefade konim
"""