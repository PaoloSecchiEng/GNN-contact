import os
import ast
import random
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from torch_geometric.nn import TransformerConv
from sklearn.model_selection import KFold
import sys


def normalize_tensor(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    return (tensor - mean) / (std + 1e-8), mean, std

def denormalize(t, m, s):
    return t * s.to(t.device) + m.to(t.device)

class TeeStdout:
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()





