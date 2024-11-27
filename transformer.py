import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncodings(nn.Module):
    
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncodings, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # (1, 2000)
        self.position = torch.arange(start=0, end=self.max_seq_length, step=1).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        
        print("position ", self.position.shape) # 1, 2000
        print("div term ", self.div_term.shape) # 64 
        
        pe = torch.zeros(self.max_seq_length, self.d_model) # 2000, 64
        
        pe[:, 0::2] = torch.sin(self.position * self.div_term)
        pe[:, 1::2] = torch.cos(self.position * self.div_term)

        self.learnable_pe = nn.Parameter(torch.randn(max_seq_length, d_model))
        self.mask_token = nn.Parameter(1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, sequence, d_model]
        seq_len = x.size(1)
        # return x + self.pe[:seq_len, :]
        return x + self.learnable_pe[:seq_len, :].unsqueeze(0)
    
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, hidden_dim):
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.d_k = self.hidden_dim // self.num_heads
        self.W_Q = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_K = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_V = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        
    
    def forward(self, x):
        # x: [B, L, HID]
        B, L = x.size(0), x.size(1)
        
        xq = self.W_Q(x) # [B, L, HID]
        xk = self.W_K(x) # [B, L, HID]
        xv = self.W_V(x) # [B, L, HID]
        
        xq = xq.view(B, L, self.num_heads, self.hidden_dim//self.num_heads) # [B, L, HEADS, HID/HEADS]
        xk = xk.view(B, L, self.num_heads, self.hidden_dim//self.num_heads) # [B, L, HEADS, HID/HEADS]
        xv = xv.view(B, L, self.num_heads, self.hidden_dim//self.num_heads) # [B, L, HEADS, HID/HEADS]
        
        xq = xq.permute(0, 2, 1, 3) # [B, H, L, D]
        xk = xk.permute(0, 2, 3, 1) # [B, H, D, L]
        xv = xv.permute(0, 2, 1, 3) # [B, H, L, D]
        
        # attention weights
        xq_xk = torch.matmul(xq, xk)
        xq_xk = xq_xk / torch.sqrt(torch.tensor(self.d_k))
        attention_weights = torch.softmax(xq_xk, dim=-1)
        print(attention_weights.shape)
        
        # apply attention
        attended = torch.einsum('bhlt,bhtd->bhld', attention_weights, xv)
        without_einsum = torch.matmul(attention_weights, xv) # (B, H, L, L) (B, H, L, D)
        print(attended.shape)
        print(without_einsum.shape)

x = torch.randn(16, 1000, 256)
pe = PositionalEncodings(256, max_seq_length=2000)
# transformer = MultiHeadAttention(num_heads=8, hidden_dim=256)
# transformer(x)
print(f"output of positional encodings: {pe(x).shape}")


# class PositionalEncodings(nn.Module):
    
#     def __init__(self, input_dim, max_sequence_length=5000):
#         super(PositionalEncodings, self).__init__()
        
#         positions = torch.arange(start=0, end=max_sequence_length).unsqueeze(1) # [5000, 1]
        
#         pe = torch.zeros(max_sequence_length, input_dim)
#         self.div_term = torch.exp(torch.arange(0, input_dim, 2) * -(math.log(10000.0) / hidden_dim))

#         pe[:, 0::2] = torch.sin(positions * self.div_term)   # evens
#         pe[:, 1::2] = torch.cos(positions * self.div_term)   # odd
        
#         self.register_buffer('pe', pe.unsqueeze(0))
        
#     def forward(self, x):
#         sequence_length = x.size(1)
#         pe_x = x + self.pe[:, :sequence_length]
#         return pe_x
    
# class TransformerEncoder(nn.Module):
    
#     def __init__(self, input_dim, hidden_dim,  num_heads):
#         super(TransformerEncoder, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
        
#         assert self.hidden_dim % num_heads == 0
        
#         self.q_w = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
#         self.k_w = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
#         self.v_w = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        
#         self.positional_encodings = PositionalEncodings(input_dim=self.input_dim, max_sequence_length=5000)
        
#     def forward(self, x):
        
#         x = self.positional_encodings(x)
        
#         Q = self.q_w(x) # B, L, HID
#         K = self.k_w(x) # B, L, HID
#         V = self.v_w(x) # B, L, HID
#         B, L, HID = Q.shape[0], Q.shape[1], Q.shape[2]
#         HEAD = self.num_heads
#         HID = HID // HEAD 
#         Q = Q.view(B, L, HEAD, HID)
#         K = K.view(B, L, HEAD, HID)
#         V = V.view(B, L, HEAD, HID)
#         attention_weights = torch.einsum("blhd,bshd->blhs", Q, K)
#         attended_x = torch.einsum("blhs,blhd->blhd", attention_weights, V)
#         attention_x = attended_x.view(B, L, HEAD * HID)
#         return attention_x
    

# if __name__ == '__main__':
        
#         sequence_length = 300
#         input_dim = 128
#         batch_size = 16 
#         hidden_dim = 256
#         x = torch.randn(batch_size, sequence_length, input_dim)
        
#         transformer = TransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=8)
        
#         out = transformer(x)
#         print(f"Transformer outputs: {out.shape}")
        