# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *


class EncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config.d_hidn, self.config.n_head, self.config.d_head)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config.d_hidn)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
    
    def forward(self, inputs, attn_mask):
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        return ffn_outputs, attn_prob


class Encoder(nn.Module):
    def __init__(self, config, n_layer):
        super().__init__()
        self.config = config
        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq + 1, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.n_layer = n_layer
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.n_layer)])
    
    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)

        outputs = self.enc_emb(inputs) + self.pos_emb(positions)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        return outputs, attn_probs



class SecondEncoder(Encoder):

    def __init__(self, config, n_layer):
        super().__init__(config, n_layer)
    
    def forward(self, inputs, cls_inputs):
        positions = torch.arange(cls_inputs.size(1), device=cls_inputs.device, dtype=cls_inputs.dtype).expand(cls_inputs.size(0), cls_inputs.size(1)).contiguous() + 1
        pos_mask = cls_inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)

        outputs = inputs + self.pos_emb(positions)
        attn_mask = get_attn_pad_mask(cls_inputs, cls_inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)

        return outputs, attn_probs


class BertEncoder(nn.Module):
    def __init__(self, bert):
        super(BertEncoder, self).__init__()
        self.bert = bert

    def gen_attention_mask(self, token_ids):
        ## masked attenion, 패딩에 패널티 부여, 학습 x
        attention_mask = token_ids.ne(1)
        return attention_mask.float()
    
    def forward(self, token_ids):
        attention_mask = self.gen_attention_mask(token_ids)
        segment_ids = torch.zeros(token_ids.size())

        output, hidden = self.bert(input_ids = token_ids.long(), token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))

        return output, hidden


class DimensionReducer(nn.Module):
    def __init__(self, input, output):
        super(DimensionReducer, self).__init__()
        self.input = input
        self.output = output
        self.fc1 = nn.Linear(self.input,(self.input + self.output)//2)
        self.fc2 = nn.Linear((self.input + self.output)//2,self.output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return x


def embedding(encoder, batch, num):
    flag = True
    for i in range(len(batch)):
        output, hidden = encoder(batch[i][:num[i]].long())
        pad_num = max_sentence_num - num[i]
        article = torch.cat([output[:, 0, :], torch.ones(pad_num,768)], dim=0).unsqueeze(0)
        if flag:
        embedded_batch = torch.cat([article, ], dim=0)
        flag=False
        else:
        embedded_batch = torch.cat([embedded_batch, article], dim=0)
    return embedded_batch
