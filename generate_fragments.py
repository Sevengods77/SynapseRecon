import os
import shutil
import math
import random
from PIL import Image

def scatter_image(image_path, rows, cols, output_path):
    """
    Breaks an image into a grid, shuffles the pieces, and places them
    on a single large 'table' canvas with random rotations without overlap.
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error: Image not found at {image_path}: {e}")
        return
    
    w, h = img.size
    patch_w = w // cols
    patch_h = h // rows
    
    fragments = []
    for r in range(rows):
        for c in range(cols):
            x_start = c * patch_w
            y_start = r * patch_h
            fragment = img.crop((x_start, y_start, x_start + patch_w, y_start + patch_h))
            if fragment.mode in ("RGBA", "P"):
                fragment = fragment.convert("RGB")
            fragments.append(fragment)
            
    random.shuffle(fragments)
    
    # Calculate cell size to safely rotate by any angle without overlapping adjacent cells
    diag = int(math.ceil(math.sqrt(patch_w**2 + patch_h**2)))
    cell_size = diag + 20 # 20px padding
    
    # Create enough cells for a loose scatter effect
    num_spots = int(math.ceil((rows * cols) * 1.5))
    grid_side = int(math.ceil(math.sqrt(num_spots)))
    
    canvas_w = grid_side * cell_size
    canvas_h = grid_side * cell_size
    
    # Create dark gray "table" background
    table_color = (40, 40, 40)
    canvas = Image.new("RGB", (canvas_w, canvas_h), table_color)
    
    # Pick random spots to place the pieces
    all_spots = [(r, c) for r in range(grid_side) for c in range(grid_side)]
    selected_spots = random.sample(all_spots, rows * cols)
    
    for fragment, spot in zip(fragments, selected_spots):
        sr, sc = spot
        # Disable rotation as per user request
        angle = 0 
        
        # Convert fragment to RGBA
        frag_rgba = fragment.convert("RGBA")
        rotated = frag_rgba.rotate(angle, expand=True)
        
        rw, rh = rotated.size
        # Center in the cell
        cx = sc * cell_size + cell_size // 2
        cy = sr * cell_size + cell_size // 2
        
        # Use rotated image as a mask for transparent background
        paste_x = cx - rw // 2
        paste_y = cy - rh // 2
        canvas.paste(rotated, (paste_x, paste_y), mask=rotated)
        
    # Create parent directories if they do not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path, "JPEG", quality=90)
    print(f"Generated single scattered image at '{output_path}'.")

def main():
    # celeba_src = r"D:\Projects\Major Project\DiffAssemble\datasets\CelebA-HQ_test"
    # wikiart_src = r"D:\Projects\Major Project\DiffAssemble\datasets\wikiart"
    input_path = r"D:\Projects\Major Project\SynapseRecon\simulated_images\original\akshay.jpg"
    base_dist = r"D:\Projects\Major Project\SynapseRecon\simulated_images"
    orig_dist = os.path.join(base_dist, "original")
    scat_dist = os.path.join(base_dist, "scattered")
    
    os.makedirs(orig_dist, exist_ok=True)
    os.makedirs(scat_dist, exist_ok=True)

    # Process only the given input_path
    fname = os.path.basename(input_path)
    out_scat = os.path.join(scat_dist, os.path.splitext(fname)[0] + ".jpg")
    scatter_image(input_path, rows=6, cols=6, output_path=out_scat)

    # # Process 10 CelebA images
    # print("Processing CelebA...")
    # try:
    #     celeba_files = [f for f in os.listdir(celeba_src) if f.endswith(('.png', '.jpg', '.jpeg'))]
    #     celeba_selected = celeba_files[:10]
    #     for f in celeba_selected:
    #         src_path = os.path.join(celeba_src, f)
    #         dest_orig = os.path.join(orig_dist, f"celeba_{f}")
    #         shutil.copy(src_path, dest_orig)
    #         
    #         # Fragment and scatter into a SINGLE image
    #         out_scat = os.path.join(scat_dist, f"celeba_{os.path.splitext(f)[0]}.jpg")
    #         scatter_image(dest_orig, rows=6, cols=6, output_path=out_scat)
    # except Exception as e:
    #     print(f"Error processing CelebA: {e}")

    # # Process 10 WikiArt images
    # print("\nProcessing WikiArt...")
    # try:
    #     wikiart_files = [f for f in os.listdir(wikiart_src) if f.endswith(('.png', '.jpg', '.jpeg'))]
    #     wikiart_files.sort()
    #     wikiart_selected = wikiart_files[:10]
    #     for f in wikiart_selected:
    #         src_path = os.path.join(wikiart_src, f)
    #         dest_orig = os.path.join(orig_dist, f"wikiart_{f}")
    #         shutil.copy(src_path, dest_orig)
    #         
    #         # Fragment and scatter into a SINGLE image
    #         out_scat = os.path.join(scat_dist, f"wikiart_{os.path.splitext(f)[0]}.jpg")
    #         scatter_image(dest_orig, rows=6, cols=6, output_path=out_scat)
    # except Exception as e:
    #     print(f"Error processing WikiArt: {e}")

if __name__ == "__main__":
    main()

