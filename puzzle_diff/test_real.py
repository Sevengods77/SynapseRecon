import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import sys
import os

# Add puzzle_diff to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__)))
from scrambled_app import DiffAssembleScrambledGradio

print("Creating mock image")
img = Image.new("RGB", (512, 512), color="black")
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 132, 132], fill="white")
draw.rectangle([200, 200, 232, 232], fill="red")
draw.rectangle([300, 300, 332, 332], fill="blue")
draw.rectangle([400, 400, 432, 432], fill="green")

dag = DiffAssembleScrambledGradio()
print("Extracting patches")
patches = dag.extract_patches(img, 4)
print("Patches shape:", patches.shape)

print("Loading model")
dag.load_model("epoch=124-step=213750.ckpt")

print("Predicting")
in_img, out_img = dag.predict(img, 2, 2)
print("Done")
