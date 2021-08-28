import torch
import torch.utils.checkpoint
from scipy import linalg
from torch import nn


class FourierFFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states, dim=-1), dim=-2).real


class FNetLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0, layer_norm_eps=0.00001):
        super().__init__()
        self.fft = FourierFFTLayer()
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        self.feed_forward = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        assert hidden_states.dim() == 3
        fft_output = self.fft(hidden_states)
        fft_output = fft_output + hidden_states
        fft_output = self.mixing_layer_norm(fft_output.transpose(1, 2)).transpose(1, 2)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = output + fft_output
        output = self.output_layer_norm(output.transpose(1, 2)).transpose(1, 2).contiguous()
        return output


class MHALayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout_rate=0, layer_norm_eps=0.00001):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate, batch_first=True)
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        self.feed_forward = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        assert hidden_states.dim() == 3
        mha_output = self.fft(hidden_states)
        mha_output = mha_output + hidden_states
        mha_output = self.mixing_layer_norm(mha_output.transpose(1, 2)).transpose(1, 2)
        intermediate_output = self.feed_forward(mha_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = output + mha_output
        output = self.output_layer_norm(output.transpose(1, 2)).transpose(1, 2).contiguous()
        return output


class FNetEncoder(nn.Module):
    def __init__(
        self, num_hidden_layers, hidden_size, intermediate_size, num_heads, dropout_rate=0, final_attention_layers=2
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.final_attention_layers = final_attention_layers
        assert num_hidden_layers >= final_attention_layers
        self.layer = nn.ModuleList(
            [
                FNetLayer(hidden_size, intermediate_size, dropout_rate)
                for _ in range(num_hidden_layers - final_attention_layers)
            ]
        )
        self.layer.extend(
            [MHALayer(hidden_size, intermediate_size, num_heads, dropout_rate) for _ in range(final_attention_layers)]
        )

    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

        return hidden_states, None
