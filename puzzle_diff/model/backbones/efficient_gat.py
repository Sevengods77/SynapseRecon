import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GraphNorm
from torch.nn import functional as F

from .resnet_equivariant import ResNet18, ResNet34, ResNet50
from .gcn import GCN
from .exophormer_gnn import Exophormer_GNN
from .Transformer_GNN import Transformer_GNN
from torchvision.transforms.functional import rotate


class Eff_GAT(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        steps,
        input_channels=2,
        output_channels=2,
        n_layers=4,
        visual_pretrained=True,
        freeze_backbone=False,
        model="efficientnet_b0",
        architecture="transformer",
        virt_nodes=4,
        all_equivariant=False
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if model == "resnet18equiv":
            self.visual_backbone = ResNet18()
        else:
            self.visual_backbone = timm.create_model(
                model, pretrained=visual_pretrained, features_only=True
            )
        self.all_equivariant=all_equivariant
        self.model = model
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.freeze_backbone = freeze_backbone

        # Determine features dimension dynamically
        dummy_patch = torch.zeros(1, 3, 32, 32)
        self.visual_backbone.eval() # Ensure eval mode for BatchNorm
        with torch.no_grad():
            dummy_feats = self.visual_features(dummy_patch)
        self.visual_backbone.train() # Back to train mode by default if not freezing
        
        # patch_feats + pos_feats (32) + time_feats (32)
        self.combined_features_dim = dummy_feats.shape[1] + 32 + 32
        print(f"DEBUG: Dynamic combined_features_dim: {self.combined_features_dim}")
        
        print(f"DEBUG: Eff_GAT initialization. model={model}, combined_features_dim={self.combined_features_dim}")

        if architecture == "transformer":
            self.gnn_backbone = Transformer_GNN(
                self.combined_features_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.combined_features_dim,
            )
        elif architecture == "gcn":
            self.gnn_backbone = GCN(
                self.combined_features_dim,
                hidden_dim=32 * 8,
                output_size=self.combined_features_dim,
            )
        elif architecture == "exophormer":
            self.gnn_backbone = Exophormer_GNN(
                self.combined_features_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.combined_features_dim,
                virt_nodes=virt_nodes
            )


        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Sequential(
            nn.Linear(input_channels, 16), nn.GELU(), nn.Linear(16, 32)
        )
        # self.GN = GraphNorm(self.combined_features_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )
        print(f"DEBUG: Eff_GAT layers. mlp[0] in_features={self.mlp[0].in_features}")

        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_channels),
        )



    def forward(self, xy_pos, time, patch_rgb, edge_index, batch):
        patch_feats = self.visual_features(patch_rgb)
        final_feats = self.forward_with_feats(
            xy_pos, time, patch_rgb, edge_index, patch_feats=patch_feats, batch=batch
        )
        return final_feats

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        patch_rgb: Tensor,
        edge_index: Tensor,
        patch_feats: Tensor,
        batch,
    ):

        time_feats = self.time_emb(time)  # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32
        # COMBINED and transform with MLP
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        
        # Check for dimension mismatch with self.mlp[0]
        if combined_feats.shape[-1] != self.mlp[0].in_features:
            print(f"CRITICAL DIMENSION MISMATCH: combined_feats input is {combined_feats.shape[-1]}, but mlp[0] expects {self.mlp[0].in_features}")
            print(f"patch_feats: {patch_feats.shape}, pos_feats: {pos_feats.shape}, time_feats: {time_feats.shape}")
        
        combined_feats = self.mlp(combined_feats)

        # GNN
        feats, attentions = self.gnn_backbone(
            x=combined_feats, edge_index=edge_index, batch=batch
        )


        # Residual + final transform
        final_feats = self.final_mlp(
            feats + combined_feats)
        return final_feats, attentions


    def visual_features(self, patch_rgb):
        patch_rgb = (patch_rgb - self.mean) / self.std

        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.visual_backbone.forward(patch_rgb)
        else:
            if self.all_equivariant:
                feats = [self.visual_backbone.forward(patch_rgb[:, i, :, :, :]) for i in range(4)]
                feats = [(feats[1][i] + feats[2][i] + feats[3][i] + feats[0][i])/4 for i in range(len(feats[1]))]

            else:
                feats = self.visual_backbone.forward(patch_rgb)
        feats = {
            "efficientnet_b0": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
            "resnet50": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
            "resnet18": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
                
            "resnet18equiv":[
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
              ]
        }[self.model]
        #if self.all_equivariant:
        #    feats[0] = self.linear1(feats[0].view(feats[0].size(0),-1)) # concatenation
        #    feats[1] = self.linear2(feats[1].view(feats[1].size(0),-1)) # concatenation features
            #feats[0] = self.linear1(feats[0])
            #feats[1] = self.linear2(feats[1])
        #    return torch.cat(feats, -1)
        #else:
        patch_feats = torch.cat(feats, -1)
        return patch_feats

