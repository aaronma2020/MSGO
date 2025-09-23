import torch
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_model, src_dim, tgt_dim):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(num_embeddings=src_dim, embedding_dim=d_model)
        self.tgt_embedding = nn.Embedding(num_embeddings=tgt_dim, embedding_dim=d_model)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=16,
                                          num_encoder_layers=12,
                                          num_decoder_layers=12,
                                          dim_feedforward=4096,
                                          dropout=0.2,
                                          batch_first=True
                                          )

        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        self.predictor = nn.Linear(d_model, tgt_dim)

    def forward(self, src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        return out

    def predict(self, out):
        return self.predictor(out)








