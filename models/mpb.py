import pickle
import torch
import copy
import torch.nn as nn
from torch.nn import ParameterList, Parameter


class MemoryPromptBank(nn.Module):
  def __init__(self):
    super().__init__()
    self.total = None
    self.pnum = None
    self.pdim = None
    self.kdim = None
    self.key_list = nn.ParameterList()
    self.prompt_list = nn.ParameterList()
    self.layer = None


  def initBank(self, layer, total, pnum, pdim, kdim, device, embedding_layer=None):
    self.layer = layer
    self.total = total
    self.pnum = pnum
    self.pdim = pdim
    self.kdim = kdim


    if len(self.key_list) > 0 or len(self.prompt_list) > 0: 
        self.key_list = nn.ParameterList()
        self.prompt_list = nn.ParameterList()


    for i in range(self.layer):
        layer_keys_params = [] 
        layer_keys_params = [nn.Parameter(torch.randn(kdim, device=device)) for j in range(total)]
        self.key_list.extend(layer_keys_params)


    for i in range(self.layer):
        layer_prompts_params = [] 
        layer_prompts_params = [nn.Parameter(torch.randn((pnum, pdim), device=device)) for j in range(total)]
        self.prompt_list.extend(layer_prompts_params) 