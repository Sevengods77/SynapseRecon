import base64
import io
import os
from io import BytesIO
from typing import List, Tuple

import cv2
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


class DiffAssembleIntegratedGradio:
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
            self.model.inference_ratio = 10
            self.model.steps = 300
            rot_status = "supports" if getattr(self.model, 'rotation', False) else "does NOT support"
            print(f"DEBUG: Model loaded. It {rot_status} rotation.")
            return f"Successfully loaded model from {ckpt_path} on {device}. Rotation support: {getattr(self.model, 'rotation', False)}"
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

    def predict_pristine(self, image, patch_dim_h, patch_dim_w):
        if self.model is None:
            return None, "Please load a model checkpoint first!"
        if image is None:
            return None, None
            
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

    def extract_patches(self, img_pil, num_expected):
        img_np = np.array(img_pil)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Threshold to find pieces (using Otsu to handle non-black backgrounds)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Optional: Clean up mask with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        print(f"DEBUG: Found {len(contours)} potential pieces in image.")
        
        pieces_tensor = []
        for cnt in contours:
            if len(pieces_tensor) >= num_expected:
                break
                
            area = cv2.contourArea(cnt)
            if area < 50: # Ignore tiny noise
                continue
                
            rect = cv2.minAreaRect(cnt)
            # Slightly inflate the box to ensure we don't cut off edges
            size = (rect[1][0] * 1.05, rect[1][1] * 1.05)
            rect = (rect[0], size, rect[2])
            
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            width = int(rect[1][0])
            height = int(rect[1][1])
            if width == 0 or height == 0:
                continue
                
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")
                                
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img_np, M, (width, height))
            
            # Use High Quality PIL resize instead of cv2.resize
            patch_pil = PILImage.fromarray(warped).resize((self._patch_size, self._patch_size), resample=Resampling.LANCZOS)
            patch_tensor = self._transforms(patch_pil)
            pieces_tensor.append(patch_tensor)
            
        while len(pieces_tensor) < num_expected:
            pieces_tensor.append(torch.zeros(3, self._patch_size, self._patch_size))
            
        patches = torch.stack(pieces_tensor)
        return patches, PILImage.fromarray(thresh)

    def puzzlize_scrambled(self, img_pil, patch_per_dim):
        num_patches = patch_per_dim[0] * patch_per_dim[1]
        patches, mask_pil = self.extract_patches(img_pil, num_patches)
        
        y = torch.linspace(-1, 1, patch_per_dim[0])
        x = torch.linspace(-1, 1, patch_per_dim[1])
        xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
        xy = einops.rearrange(xy, "x y c -> (x y) c")

        adj_mat = torch.ones(num_patches, num_patches)
        edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
        
        data = pyg_data.Data(
            x=xy,
            patches=patches,
            edge_index=edge_index,
            patches_dim=torch.tensor([patch_per_dim]),
            batch=torch.zeros(xy.shape[0], dtype=torch.long),
        )
        return data, mask_pil

    def predict_scrambled(self, image, patch_dim_h, patch_dim_w):
        if self.model is None:
            return None, None, "Please load a model checkpoint first!"
        if image is None:
            return None, None, None
            
        patch_per_dim = [int(patch_dim_h), int(patch_dim_w)]
        try:
            graph_in, mask_pil = self.puzzlize_scrambled(image, patch_per_dim)
            graph_in = graph_in.to(self.model.device)
            
            with torch.no_grad():
                imgs, attentions = self.model.prediction_step(graph_in, 0)

            input_image = self.create_image_from_patches(
                graph_in.patches, graph_in.x, n_patches=patch_per_dim
            ).resize((256, 256))

            pred_res = self.create_image_from_patches(
                graph_in.patches, imgs[-1], n_patches=patch_per_dim
            ).resize((256, 256))

            return mask_pil, input_image, pred_res
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise gr.Error(f"Error processing image: {str(e)}")


def main():
    dag = DiffAssembleIntegratedGradio()
    
    with gr.Blocks(title="DiffAssemble Integrated Solver") as demo:
        gr.Markdown("# DiffAssemble: Complete Puzzle Solver")
        
        with gr.Row():
            ckpt_input = gr.Textbox(label="Checkpoint Path (.ckpt)", placeholder="path/to/last.ckpt", value="epoch=124-step=213750.ckpt")
            load_btn = gr.Button("Load Model")
            load_status = gr.Label(value="No model loaded")
            
        load_btn.click(fn=dag.load_model, inputs=ckpt_input, outputs=load_status)
        
        with gr.Tabs():
            with gr.Tab("Pristine Image -> Shatter & Solve"):
                with gr.Row():
                    with gr.Column():
                        patch_h1 = gr.Number(label="Patches Height", value=12, precision=0)
                        patch_w1 = gr.Number(label="Patches Width", value=12, precision=0)
                        image_input1 = gr.Image(type="pil", label="Upload Pristine Image")
                        solve_btn1 = gr.Button("Shatter and Solve")
                    
                    with gr.Column():
                        puzzle_output1 = gr.Image(type="pil", label="Scrambled Input")
                        solved_output1 = gr.Image(type="pil", label="Assembled Result")
                
                solve_btn1.click(
                    fn=dag.predict_pristine, 
                    inputs=[image_input1, patch_h1, patch_w1], 
                    outputs=[puzzle_output1, solved_output1]
                )
                
            with gr.Tab("Scattered Image -> Extract & Solve"):
                gr.Markdown("Upload an image with scattered puzzle pieces on a dark background. The app will extract the pieces and ask the model to assemble them.")
                with gr.Row():
                    with gr.Column():
                        patch_h2 = gr.Number(label="Grid Height (e.g. 6)", value=6, precision=0)
                        patch_w2 = gr.Number(label="Grid Width (e.g. 6)", value=6, precision=0)
                        image_input2 = gr.Image(type="pil", label="Upload Scattered Image")
                        solve_btn2 = gr.Button("Extract and Solve")
                    
                    with gr.Column():
                        mask_output2 = gr.Image(type="pil", label="Detection Mask (Debug)")
                        puzzle_output2 = gr.Image(type="pil", label="Extracted Pieces")
                        solved_output2 = gr.Image(type="pil", label="Assembled Result")
                
                solve_btn2.click(
                    fn=dag.predict_scrambled, 
                    inputs=[image_input2, patch_h2, patch_w2], 
                    outputs=[mask_output2, puzzle_output2, solved_output2]
                )
        
    demo.launch(server_name="127.0.0.1", server_port=7862)


if __name__ == "__main__":
    main()
