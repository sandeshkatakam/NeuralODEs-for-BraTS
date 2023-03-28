#########################################
#                                       #
#                                       #
#       NeuralODE U-NET Model           #
#                                       #
#                                       #
#########################################

# Installing dependencies

from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdym import *
from torchdyn import RungeKutta4

%load_ext autoreload
%autoreload 2

import torch 
import torch.utils.data as data

model = NeuralODE(f, sensitivity='adjoint', solver='rk4').to(device)

class U_Net(nn.Module):
    def __init__(self, ):

