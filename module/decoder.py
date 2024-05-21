import torch.nn as nn
import torch
from einops import rearrange, repeat


class TransformerDecoder(nn.Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        encoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=4, dim_feedforward=1024)
        self.cls = nn.Parameter(torch.randn(1, 1, 768))
        self.encoder1 = nn.TransformerDecoder(encoder_layer, num_layers=1)
        self.encoder2 = nn.TransformerDecoder(encoder_layer, num_layers=1)
        self.encoder3 = nn.TransformerDecoder(encoder_layer, num_layers=1)
        self.encoder4 = nn.TransformerDecoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(768, 2)

    def forward(self, x, context):
        b, n, _ = x.shape
        # cls_tokens = repeat(self.cls, '() n d -> b n d', b=b)
        x = self.encoder1(x, context)  # x-Q, context-K,V from last decoder
        x = self.encoder2(x, context)
        x = self.encoder3(x, context)
        x = self.encoder4(x, context)
        out = self.fc(x)
        out = out.mean(1)

        return out
