# Monkey patch torch_scatter for Windows compatibility BEFORE any other imports
try:
    import torch
    import torch_scatter
    
    def native_scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
        if reduce == "sum" or reduce == "add":
            if out is None:
                if dim_size is None:
                    dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
                out_shape = list(src.shape)
                out_shape[dim] = dim_size
                out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src) if src.dim() > index.dim() else index, src)
        elif reduce == "mean":
            sum_val = native_scatter(src, index, dim, dim_size=dim_size, reduce="sum")
            counts = torch.zeros(sum_val.shape[dim], dtype=src.dtype, device=src.device)
            ones = torch.ones(index.shape, dtype=src.dtype, device=src.device)
            counts.scatter_add_(0, index, ones)
            if sum_val.dim() > 1:
                view_shape = [1] * sum_val.dim()
                view_shape[dim] = -1
                counts = counts.view(view_shape)
            return sum_val / counts.clamp(min=1)
        else:
            return torch_scatter.scatter(src, index, dim, out, dim_size, reduce)

    # Apply the patch
    torch_scatter.scatter = native_scatter
    print("Applied native scatter monkey patch for Windows compatibility")
except ImportError:
    pass

import argparse
import os
import random
import string
import sys

import torch
import torch_geometric
import torch.optim.lr_scheduler

# Compatibility fix for older torch versions where LRScheduler is not defined
if not hasattr(torch.optim.lr_scheduler, "LRScheduler"):
    torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler._LRScheduler

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import argparse
import math
import random
import string

import pytorch_lightning as pl
from dataset import dataset_utils as du
from model import spatial_diffusion as sd
from model import spatial_diffusion_discrete as sdd
from model import spatial_diffusion_discrete_rot as sdd_rot
from model.spatial_diffusion import GNN_Diffusion
from model.spatial_diffusion_discrete import GNN_Diffusion_Discrete
from model.spatial_diffusion_discrete_rot import GNN_Diffusion_Discrete_ROT

import matplotlib
import dataloader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    # Set seed
    pl.seed_everything(42)

    # ... rest of the file ...
    # Wait, I should probably use replace_file_content or at least read the whole file first
    # to avoid overwriting the logic.
    # I already have the first 60 lines. 
    # Let's just use replace_file_content to move the block.
