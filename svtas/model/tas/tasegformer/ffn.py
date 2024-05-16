import torch
import torch.nn as nn

class ResdualMLPBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 dropout=0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, 1),
                nn.Conv1d(embed_dim, embed_dim, 1),
                nn.SiLU())
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
    def forward(self, x, masks):
        out = self.mlp(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return (x + out) * masks[:, 0:1, :]

class ResdualDilationConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 dropout=0.0) -> None:
        super().__init__()
        self.dialtion_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.SiLU()
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
    def forward(self, x, masks):
        out = self.dialtion_conv(x)
        out = self.conv_1x1(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return (x + out) * masks[:, 0:1, :]