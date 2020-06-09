import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import hparams as hp
import numpy as np
import modules
import random
import utils


class _ReferenceEncoder(nn.Module):
    def __init__(self):
        super(_ReferenceEncoder, self).__init__()

        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(1, 1),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hp.num_mels, 3, 1, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.encoder_dim // 2,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, hp.num_mels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)

        self.gru.flatten_parameters()
        out, _ = self.gru(out)

        return out

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class LST(nn.Module):
    """ Local Style Token """

    def __init__(self):
        super(LST, self).__init__()

        self.style_embedding = nn.Parameter(
            torch.FloatTensor(hp.token_num, hp.style_dim))
        nn.init.normal_(self.style_embedding, mean=0, std=0.5)

        self.encoder = _ReferenceEncoder()
        self.conv = modules.BatchNormConv1d(hp.encoder_dim // 2,
                                            hp.encoder_dim // 2,
                                            hp.kernel_size,
                                            hp.stride,
                                            hp.padding)

        self.linear_1 = nn.Linear(hp.encoder_dim // 2, hp.encoder_dim // 4)
        self.linear_2 = nn.Linear(hp.encoder_dim // 4, hp.token_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def max_core(self, input, base=16.0, eps=1e-6):
        out = torch.pow(base, input)
        div = (torch.sum(out, -1)+eps).unsqueeze(-1)
        out = torch.div(out, div)
        return out

    def pooling_core(self, encoder_output, duration, src_pos):
        result = list()
        for batch_ind in range(src_pos.size(0)):
            cur_index = 0
            sq_len = torch.max(src_pos[batch_ind]).item()
            one_batch = list()
            for i in range(sq_len):
                cut_len = duration[batch_ind][i]
                if cut_len == 0:
                    one_batch.append(one_batch[-1].zero_())
                else:
                    one_batch.append(torch.sum(
                        encoder_output[batch_ind][cur_index:cur_index+cut_len], 0)/cut_len.float())
                cur_index += duration[batch_ind][i]
            one_batch = torch.stack(one_batch)
            result.append(one_batch)
        output = utils.pad(result, mel_max_length=src_pos.size(-1))
        return output

    def forward(self, mel, duration, src_pos):
        encoder_output = self.encoder(mel)

        pooling_output = self.pooling_core(encoder_output, duration, src_pos)
        pooling_output = self.conv(pooling_output.contiguous(
        ).transpose(1, 2)).contiguous().transpose(1, 2)

        linear_output = self.relu(self.linear_1(pooling_output))
        linear_output = self.softmax(self.linear_2(linear_output))
        lst_output = torch.matmul(linear_output, self.style_embedding)

        return lst_output, encoder_output

    def infer(self, style_index, length):
        if False:
            style_embeddings = list()
            for _ in range(length):
                style_embeddings.append(
                    self.style_embedding[random.randint(0, hp.token_num-1)])
            return torch.stack([torch.stack(style_embeddings)]).float().cuda()
        else:
            return torch.stack([self.style_embedding[style_index].unsqueeze(0).expand(length, -1)])


class GST(nn.Module):

    def __init__(self):

        super().__init__()
        self.encoder = ReferenceEncoder()
        self.stl = STL()

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)

        return style_embed


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self):
        super(ReferenceEncoder, self).__init__()

        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hp.num_mels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.style_dim // 2,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, hp.num_mels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        _, out = self.gru(out)  # out --- [1, N, E]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self):
        super(STL, self).__init__()

        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num,
                                                    hp.style_dim // hp.num_heads))
        d_q = hp.style_dim // 2
        d_k = hp.style_dim // hp.num_heads
        self.attention = MultiHeadAttention(query_dim=d_q,
                                            key_dim=d_k,
                                            num_units=hp.style_dim,
                                            num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = torch.tanh(self.embed).unsqueeze(0).expand(
            N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(
            in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim,
                               out_features=num_units, bias=False)
        self.W_value = nn.Linear(
            in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        # [h, N, T_q, num_units/h]
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0),
                        dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
