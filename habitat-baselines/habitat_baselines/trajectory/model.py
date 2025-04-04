import torch.nn as nn
import torch
import math
# import Optional
from typing import Optional, Tuple
from torch import Tensor
# import F
from torch.nn import functional as F
import copy

def _get_clone(module):
    return copy.deepcopy(module)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.mlp(x) # [nenvs, output_dim]

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TrajectoryTransformer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.positional_encoding = PositionalEncoding(self.hidden_dim, self.dropout)
        #encoder_layers = TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads, dropout=self.dropout, batch_first=True)
        encoder_layers = TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dim_feedforward=self.hidden_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        self.input_layer = nn.Linear(input_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim*time_dim, self.output_dim)


    def forward(self, x,
                mask=None,
                ):
        # ! Doubts: Do we need a Relu at the end, if yes why? 
        # ! If you reshape your input to be (batch_size, sequence_length*feature_dimension), the Transformer model would interpret this as having sequence_length*feature_dimension time steps in each sequence, each with a single feature, which is not what you want.
        # x = x.view(x.shape[0], -1, self.input_dim)
        # Create a mask where all values are 0
        
        # Create mask env,seq,seq
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(x.device)
        # for each value in x, set the mask to 1 if the value is 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i][j][0] == 0:
                    mask[i][j] = 1

        # Repeat the mask for each head
        mask = mask.repeat(self.num_heads, 1, 1).reshape(self.num_heads*x.shape[0], x.shape[1], x.shape[1])

        x = x.permute(1,0,2)
        x = self.input_layer(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask).permute(1,0,2)

        x = self.output_layer(x.reshape(x.shape[0], -1))

        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
  
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output 

class AdaptationModule(nn.Module):
    def __init__(self, input_dim, time_dim=20, hidden_dim=128, output_dim=128):
        super().__init__()
        self.mlp_space = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(True))
        self.mlp_time = nn.Sequential(
            nn.Linear(time_dim, 5),
            nn.ReLU(True),
            nn.Linear(5, 1),
            nn.ReLU(True))
        self.linear_output = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.mlp_space(x).permute(0,2,1)
        x = self.mlp_time(x).permute(0,2,1)
        x = self.linear_output(x)
        return x
    
if __name__ == "__main__":
    import torch
    hidden_dim = 128
    output_dim = 128
    time_dim = 20
    n_env = 2
    pos_var = 4 # x,y,z,orientation
    #x = torch.rand(n_env, time_dim*pos_var)
    # x = x.reshape(n_env, -1)
    #print(x.shape)
    #x = x.view(n_env, time_dim, pos_var)
    x = torch.rand(n_env, time_dim-5, pos_var)
    to_cat  = torch.zeros(n_env, 5, pos_var)
    x = torch.cat((to_cat, x), dim=1) 
    x[1] = 0
    
    encoder = TrajectoryTransformer(input_dim=pos_var, hidden_dim=32, output_dim=output_dim, time_dim=time_dim)
    x = encoder(x) # S, N, E
    print(x.shape)