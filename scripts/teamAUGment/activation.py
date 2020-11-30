import warnings
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn import functional as F

  
class AReLU(Module):

   
    def __init__(self, alpha = 0.90, beta = 2.0):
        super().__init__()
 
        self.alphap = Parameter(torch.tensor([alpha]),requires_grad=True)
        self.betap = Parameter(torch.tensor([beta]),requires_grad=True)
        

    def forward(self, input):
    
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        
        alpha = torch.clamp(self.alphap, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.betap)
        
        return F.relu(input) * beta.to(device) - F.relu(-input) * alpha.to(device)

