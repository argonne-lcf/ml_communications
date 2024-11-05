from mpi4py import MPI
# comm = MPI.COMM_WORLD
# comm
# RANK = MPI
# dev = 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class DistributedMLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        def FFNN(emb_dim=emb_dim):
            ffnn_layer = nn.Sequential(
                Linear(emb_dim, 4*emb_dim),
                nn.GELU(),
                Linear(4*emb_dim, emb_dim),
            )
            return ffnn_layer
        
        self.FFNN1 = FFNN()
        self.FFNN2 = FFNN()
        
    def forward(self, x, SP=False):
        B, s, hc, hs = x.size()
        x = x.reshape(B, s, hc*hs)
        x = self.FFNN1(x)
        x = self.FFNN2(x)
        x = x.reshape(B, s, hc, hs)
        return x

if __name__ == "__main__":
    ## TODO: add argparse
    B = 1
    s = 128
    hc = 16
    hs = 128
    eps = 1e-8 ## Stingency of gradient and output accuracy. 

    x = torch.randn(B, s, hc, hs)
    label = torch.randn(B, s, hc, hs)
    emb_dim = hc * hs
    
    ## TODO: add if statement for unit-test
    def empty_grad(model):
        for param in model.parameters():
            param.grad = None
    no_ulysses = DistributedMLP(emb_dim)
    ulysses = DistributedMLP(emb_dim)
    ulysses.load_state_dict(no_ulysses.state_dict()) ## keep the params the same

    no_ulysses_out = no_ulysses(x)
    ulysses_out = ulysses(x)

    no_ulysses_loss = F.mse_loss(no_ulysses_out, label)
    ulysses_loss = F.mse_loss(ulysses_out, label)

    no_ulysses_loss.backward()
    ulysses_loss.backward()
    # no_ulysses_out.backward(x) ## Interesting use of backward. This is called Jacobian-vector product, 
    # which is mathematically equivalent to summing the output vector into a scalar and computing the gradient 
    # w.r.t. to the input vector (I think).

    out_diff = torch.norm(no_ulysses_out - ulysses_out, p=1) ## l1 norm
    total_grad_diff = 0
    for grad1, grad2 in zip(no_ulysses.parameters(), ulysses.parameters()):
        print(f"grad1: {grad1.shape}")
        print(f"grad2: {grad2.shape}")
        total_grad_diff += torch.norm(grad1 - grad2, p=1)
    num_parameters = sum([param.numel() for param in ulysses.parameters()])
    avg_grad_diff_per_param = total_grad_diff / num_parameters

    print(f"out_diff: {out_diff}")
    print(f"total_grad_diff: {total_grad_diff}")
    print(f"avg_grad_diff_per_param: {avg_grad_diff_per_param}")

    assert out_diff < eps
    assert avg_grad_diff_per_param < eps

#     ## TODO: implement Ulysses across ranks
#     ## TODO: figure out why this takes so goddamn long. 
# ## mpiexec 