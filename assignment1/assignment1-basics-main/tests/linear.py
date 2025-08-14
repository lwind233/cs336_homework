import torch
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        self.weights= torch.nn.Parameter(torch.randn(out_features,in_features,dtype=dtype,device=device))
        torch.nn.init.trunc_normal_(self.weights)

    def forward(self,x):
        out = einsum(x,self.weights, '... d_in, d_out d_in -> ... d_out')
        return out
    