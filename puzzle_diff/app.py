import base64
import io
import os
from io import BytesIO
from typing import List, Tuple

import einops
import gradio as gr
import numpy as np
import requests
import skimage
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torchvision
import torchvision.transforms as transforms
from model import spatial_diffusion_on_angle as sd
from PIL import Image as PILImage
from PIL.Image import Resampling
from torch import Tensor


def encode(image) -> str:
    # convert image to bytes
    with BytesIO() as output_bytes:
        PIL_image = PILImage.fromarray(skimage.img_as_ubyte(image))
        PIL_image.save(output_bytes, "JPEG")
        bytes_data = output_bytes.getvalue()

    # encode bytes to base64 string
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return base64_str


@torch.jit.script
def divide_images_into_patches(
    img, patch_per_dim: List[int], patch_size: int
) -> Tuple[Tensor, Tensor]:
    # divide images in non-overlapping patches based on patch size
    img2 = img.permute(1, 2, 0)
    patches = img2.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    y = torch.linspace(-1, 1, patch_per_dim[0])
    x = torch.linspace(-1, 1, patch_per_dim[1])
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)

    return xy, patches


class DiffAssembleGradio:
    def __init__(self):
        self.model = None
        self._patch_size = 32
        self._transforms = transforms.Compose([transforms.ToTensor()])

    def load_model(self, ckpt_path):
        if not ckpt_path or not os.path.exists(ckpt_path):
            return f"Error: Checkpoint file not found at {ckpt_path}"
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = sd.GNN_Diffusion.load_from_checkpoint(ckpt_path)
            self.model.to(device)
            self.model.eval()
            self.model.noise_weight = 1
            return f"Successfully loaded model from {ckpt_path} on {device}"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def create_image_from_patches(self, patches, pos, n_patches):
        patch_size = self._patch_size
        height = patch_size * n_patches[0]
        width = patch_size * n_patches[1]
        new_image = PILImage.new("RGB", (width, height))
        for p in range(patches.shape[0]):
            patch = patches[p, :]
            patch = PILImage.fromarray(
                ((patch.permute(1, 2, 0)) * 255).cpu().numpy().astype(np.uint8)
            )

            x = pos[p, 0] * (1 - 1 / n_patches[0])
            y = pos[p, 1] * (1 - 1 / n_patches[1])
            x_pos = int((x + 1) * width / 2) - patch_size // 2
            y_pos = int((y + 1) * height / 2) - patch_size // 2
            new_image.paste(patch, (x_pos, y_pos))
        return new_image

    def puzzlize(self, img, patch_per_dim):
        height = patch_per_dim[0] * self._patch_size
        width = patch_per_dim[1] * self._patch_size
        img = img.resize((width, height), resample=Resampling.BICUBIC)
        img = self._transforms(img)

        xy, patches = divide_images_into_patches(img, patch_per_dim, self._patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        adj_mat = torch.ones(
            patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
        )
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)
        data = pyg_data.Data(
            x=xy,
            patches=patches,
            edge_index=edge_index,
            patches_dim=torch.tensor([patch_per_dim]),
            batch=torch.zeros(xy.shape[0], dtype=torch.long),
        )
        return data

    def predict(self, image, patch_dim_h, patch_dim_w):
        if self.model is None:
            return None, "Please load a model checkpoint first!"
            
        patch_per_dim = [int(patch_dim_h), int(patch_dim_w)]
        graph_in = self.puzzlize(image, patch_per_dim)
        graph_in = graph_in.to(self.model.device)
        
        with torch.no_grad():
            imgs, attentions = self.model.prediction_step(graph_in, 0)

        input_image = self.create_image_from_patches(
            graph_in.patches, imgs[0], n_patches=patch_per_dim
        ).resize((256, 256))

        pred_res = self.create_image_from_patches(
            graph_in.patches, imgs[-1], n_patches=patch_per_dim
        ).resize((256, 256))

        return input_image, pred_res


def main():
    dag = DiffAssembleGradio()
    
    with gr.Blocks(title="DiffAssemble Interactive Puzzle Solver") as demo:
        gr.Markdown("# DiffAssemble: Solving Puzzles with Diffusion")
        
        with gr.Row():
            ckpt_input = gr.Textbox(label="Checkpoint Path (.ckpt)", placeholder="path/to/last.ckpt", value="epoch=124-step=213750.ckpt")
            load_btn = gr.Button("Load Model")
            load_status = gr.Label(value="No model loaded")
            
        load_btn.click(fn=dag.load_model, inputs=ckpt_input, outputs=load_status)
        
        with gr.Row():
            with gr.Column():
                patch_h = gr.Number(label="Patches Height", value=12, precision=0)
                patch_w = gr.Number(label="Patches Width", value=12, precision=0)
                image_input = gr.Image(type="pil", label="Upload Image")
                solve_btn = gr.Button("Shatter and Solve")
            
            with gr.Column():
                puzzle_output = gr.Image(type="pil", label="Scrambled Input")
                solved_output = gr.Image(type="pil", label="Assembled Result")
        
        solve_btn.click(
            fn=dag.predict, 
            inputs=[image_input, patch_h, patch_w], 
            outputs=[puzzle_output, solved_output]
        )
        
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
