from json.tool import main
from operator import imod
from tkinter import Y
import numpy as np
import pandas as pd
import torch

from utils import *

class MLP(torch.nn.Module):
    def __init__(self, num_i, num_h, num_o=1):
        super(MLP,self).__init__()
        
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_o)
        self.output = torch.nn.Sigmoid()
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.output(x)
        return x
