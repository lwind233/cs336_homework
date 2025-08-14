import torch
from torch import einsum

class Linear(torch.nn.Module):
    def __init__(self,in_features,out_features,device,dtype):
        self.w = torch.nn.Parameter(torch.randn(out_features,in_features,dtype=dtype,device=device))

    def forward(self,x):
        out = einsum(x,self.w, '... dim , out_dim in_dim')
        return out
    
