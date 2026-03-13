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

import math
import random
import string

import pytorch_lightning as pl
from dataset import dataset_utils as du
from model import spatial_diffusion as sd
from model import spatial_diffusion_discrete as sdd
from model import spatial_diffusion_discrete_rot as sdd_rot
from model.spatial_diffusion import GNN_Diffusion
from model.spatial_diffusion_discrete import GNN_Diffusion as GNN_Diffusion_Discrete
from model.spatial_diffusion_discrete_rot import GNN_Diffusion as GNN_Diffusion_Discrete_ROT

from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

import wandb
import logging

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str

def main(
    batch_size,
    gpus,
    steps,
    num_workers,
    dataset,
    puzzle_sizes,
    sampling,
    inference_ratio,
    offline,
    noise_weight,
    checkpoint_path,
    discrete,
    rotation,
):
    ### Define dataset

    if rotation:
        _, test_dt, puzzle_sizes = du.get_dataset_ROT(
            dataset=dataset,
            puzzle_sizes=puzzle_sizes,
        )
    else:
        _, test_dt, puzzle_sizes = du.get_dataset(
            dataset=dataset, puzzle_sizes=puzzle_sizes
        )

    dl_test = torch_geometric.loader.DataLoader(
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    if discrete and rotation:
        model_cls = sdd_rot.GNN_Diffusion_Discrete_ROT
    elif discrete:
        model_cls = sdd.GNN_Diffusion_Discrete
    else:
        model_cls = sd.GNN_Diffusion

    model = model_cls.load_from_checkpoint(checkpoint_path)
    model.noise_weight = noise_weight
    model.inference_ratio = inference_ratio
    model.initialize_torchmetrics(puzzle_sizes)
    model.steps = steps
    ### define training

    franklin = True if gpus > 1 else False

    experiment_name = f"eval-{dataset}-{puzzle_sizes}-{steps}-{get_random_string(6)}"

    tags = [f"{dataset}", f'{"franklin" if franklin else "fisso"}', "train"]

    wandb_logger = WandbLogger(
        project="Puzzle-Diff",
        name=experiment_name,
        tags=tags,
        offline=offline,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=[ModelSummary(max_depth=2)],
    )
    logging.warning(f"Saving to {experiment_name}")
    trainer.predict(model, dl_test)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--dataset", default="wikiart", choices=["celeba", "wikiart", "cifar100"]
    )
    ap.add_argument("--sampling", default="DDIM", choices=["DDPM", "DDIM"])
    ap.add_argument("--inference_ratio", type=int, default=10)
    ap.add_argument(
        "--puzzle_sizes",
        nargs="+",
        default=[12],
        type=int,
        help="Input a list of values",
    )
    ap.add_argument("--offline", type=str2bool, default=False)
    ap.add_argument("--noise_weight", type=float, default=1.0)
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--discrete", type=str2bool, default=False)
    ap.add_argument("--rotation", type=str2bool, default=False)

    args = ap.parse_args()
    print(args)
    main(
        batch_size=args.batch_size,
        gpus=args.gpus,
        steps=args.steps,
        num_workers=args.num_workers,
        dataset=args.dataset,
        puzzle_sizes=args.puzzle_sizes,
        sampling=args.sampling,
        inference_ratio=args.inference_ratio,
        offline=args.offline,
        noise_weight=args.noise_weight,
        checkpoint_path=args.checkpoint_path,
        discrete=args.discrete,
        rotation=args.rotation,
    )
