import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncodings(nn.Module):
    
    def __init__(self, input_dim, max_sequence_length=5000):
        super(PositionalEncodings, self).__init__()
        
        positions = torch.arange(start=0, end=max_sequence_length).unsqueeze(1) # [5000, 1]
        
        pe = torch.zeros(max_sequence_length, input_dim)
        self.div_term = torch.exp(torch.arange(0, input_dim, 2) * -(math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(positions * self.div_term)   # evens
        pe[:, 1::2] = torch.cos(positions * self.div_term)   # odd
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        sequence_length = x.size(1)
        pe_x = x + self.pe[:, :sequence_length]
        return pe_x
    
class TransformerEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim,  num_heads):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        assert self.hidden_dim % num_heads == 0
        
        self.q_w = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.k_w = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.v_w = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        
        self.positional_encodings = PositionalEncodings(input_dim=self.input_dim, max_sequence_length=5000)
        
    def forward(self, x):
        
        x = self.positional_encodings(x)
        
        Q = self.q_w(x) # B, L, HID
        K = self.k_w(x) # B, L, HID
        V = self.v_w(x) # B, L, HID
        B, L, HID = Q.shape[0], Q.shape[1], Q.shape[2]
        HEAD = self.num_heads
        HID = HID // HEAD 
        Q = Q.view(B, L, HEAD, HID)
        K = K.view(B, L, HEAD, HID)
        V = V.view(B, L, HEAD, HID)
        attention_weights = torch.einsum("blhd,bshd->blhs", Q, K)
        attended_x = torch.einsum("blhs,blhd->blhd", attention_weights, V)
        attention_x = attended_x.view(B, L, HEAD * HID)
        return attention_x
    
    

if __name__ == '__main__':
        
        sequence_length = 300
        input_dim = 128
        batch_size = 16 
        hidden_dim = 256
        x = torch.randn(batch_size, sequence_length, input_dim)
        
        transformer = TransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=8)
        
        out = transformer(x)
        print(f"Transformer outputs: {out.shape}")
        
        