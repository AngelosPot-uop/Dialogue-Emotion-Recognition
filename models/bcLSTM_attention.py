import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1. / (hidden_dim ** 0.5)

    def forward(self, query, key, value, mask=None):
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, value)
        return context, attn_weights


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate,
                            bidirectional=True)
        self.attention = ScaledDotProductAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # print(f"Output shape after LSTM: {output.shape}")

        # Using the output of the LSTM as both key and value
        context, attn_weights = self.attention(output, output, output)
        # print(f"Context vector shape: {context.shape}")

        context = self.dropout(context)
        out = self.fc(context)
        # print(f"Final output shape: {out.shape}")

        return out

