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
    
class Embedding(torch.nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None,dtype=None):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(num_embeddings,embedding_dim,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.weights)

    def forward(self,x):
        b = x.shape[0]
        s = x.shape[1]
        x = rearrange(x,'b s -> (b s)')
        out = self.weights[x]
        out = rearrange(out,'(b s) d -> b s d', b = b,s = s)
        return out
    
class RMSnorm(torch.nn.Module):
    def __init__(self,d_model,eps,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.gain = torch.nn.Parameter(torch.randn(d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.gain)
        self.eps = eps

    def forward(self,x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        b = x.shape[0]
        s = x.shape[1]
        x = rearrange(x, 'b s d -> (b s) d')
        rms_x = (x * x).sum(-1).unsqueeze(1)
        rms_x = (rms_x / self.d_model + self.eps) ** 0.5

        output = x / rms_x * self.gain
        output = rearrange(output, '(b s) d -> b s d',b = b, s = s)
        output.to(in_dtype)
        return output


    