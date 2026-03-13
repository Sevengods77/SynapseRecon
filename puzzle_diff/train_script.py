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
import sys
import glob
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
from dataset.dataset_utils import get_dataset, get_dataset_ROT

# Import model components
from model import spatial_diffusion_on_angle as sd_angle
from model.spatial_diffusion_on_angle import ModelMeanType

import matplotlib
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main(args):
    # Set seed
    pl.seed_everything(42)

    # Get dataset
    if args.rotation:
        train_dataset, val_dataset, _ = get_dataset_ROT(
            args.dataset,
            args.puzzle_sizes,
            args.data_augmentation,
            args.angle_type,
            args.degree,
            args.unique_graph,
            args.all_equivariant,
            args.random_dropout
        )
    else:
        train_dataset, val_dataset, _ = get_dataset(
            args.dataset,
            args.puzzle_sizes,
            args.data_augmentation,
            args.degree,
            args.unique_graph,
        )

    # Data loaders
    train_loader = torch_geometric.loader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Logger
    wandb_logger = WandbLogger(
        project="DiffAssemble",
        name=f"{args.dataset}_{args.puzzle_sizes}",
        id=args.wandb_id,
        offline=args.offline,
    )

    # Model - Using correct class name GNN_Diffusion and ONLY supported arguments
    model = sd_angle.GNN_Diffusion(
        steps=args.steps,
        sampling=args.sampling,
        inference_ratio=args.inference_ratio,
        virt_nodes=args.virt_nodes,
        n_layers=args.n_layers,
        classifier_free_w=args.classifier_free_w,
        classifier_free_prob=args.classifier_free_prob,
        noise_weight=args.noise_weight,
        model_mean_type=ModelMeanType.START_X if args.predict_xstart else ModelMeanType.EPSILON,
        rotation=args.rotation,
        angle_type=args.angle_type,
        freeze_backbone=args.freeze_backbone,
        visual_pretrained=args.visual_pretrained,
        backbone=args.backbone,
        architecture=args.architecture,
        all_equivariant=args.all_equivariant,
        puzzle_sizes=args.puzzle_sizes,
    )
    print("Model Architecture:")
    print(model)

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.dataset}_{args.puzzle_sizes}",
        filename="{epoch}-{overall_acc:.4f}",
        save_top_k=5,
        monitor="overall_acc",
        mode="max",
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.acc_grad if args.acc_grad > 0 else 1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
    )

    # Train
    if not args.evaluate:
        if args.checkpoint_path != "":
            trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint_path)
        else:
            trainer.fit(model, train_loader, val_loader)
    else:
        trainer.validate(model, val_loader, ckpt_path=args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="celeba")
    parser.add_argument("--sampling", type=str, default="DDIM")
    parser.add_argument("--inference_ratio", type=int, default=10)
    parser.add_argument("--degree", type=str, default="100%")
    parser.add_argument("--virt_nodes", type=int, default=4)
    parser.add_argument("--unique_graph", type=str2bool, default=False)
    parser.add_argument("--inf_fully", type=str2bool, default=False)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--puzzle_sizes", nargs="+", type=int, default=[6])
    parser.add_argument("--offline", type=str2bool, default=False)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--classifier_free_w", type=float, default=0.2)
    parser.add_argument("--classifier_free_prob", type=float, default=0.0)
    parser.add_argument("--data_augmentation", type=str, default="none")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--noise_weight", type=float, default=0.0)
    parser.add_argument("--predict_xstart", type=str2bool, default=False)
    parser.add_argument("--rotation", type=str2bool, default=False)
    parser.add_argument("--only_rotation", type=str2bool, default=False)
    parser.add_argument("--angle_type", type=str, default="radian")
    parser.add_argument("--freeze_backbone", type=str2bool, default=False)
    parser.add_argument("--visual_pretrained", type=str2bool, default=True)
    parser.add_argument("--discrete", type=str2bool, default=False)
    parser.add_argument("--cold_diffusion", type=str2bool, default=False)
    parser.add_argument("--loss_type", type=str, default="cross_entropy")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--architecture", type=str, default="transformer")
    parser.add_argument("--all_equivariant", type=str2bool, default=False)
    parser.add_argument("--evaluate", type=str2bool, default=False)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--acc_grad", type=int, default=0)
    parser.add_argument("--missing", type=int, default=0)
    parser.add_argument("--random_dropout", type=str2bool, default=False)
    parser.add_argument("--save_eval_images", type=str2bool, default=False)
    args = parser.parse_args()

    main(args)
