import os
import shutil
import math
import random
from PIL import Image

def get_best_grid(w, h, min_f=8, max_f=12):
    """
    Finds the best (rows, cols) pair such that rows * cols is between min_f and max_f,
    and the aspect ratio of the grid (cols/rows) is closest to the image aspect ratio (w/h).
    """
    target_ratio = w / h
    best_r, best_c = 1, min_f
    min_diff = float('inf')
    
    # Try all combinations of r and c that result in total between min_f and max_f
    for total in range(min_f, max_f + 1):
        for r in range(1, total + 1):
            if total % r == 0:
                c = total // r
                current_ratio = c / r
                diff = abs(current_ratio - target_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_r, best_c = r, c
                    
    return best_r, best_c

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
        # Disable rotation as per previous version's configuration
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
    print(f"Generated scattered image ({rows}x{cols} = {rows*cols} pieces) at '{output_path}'.")

def main():
    base_dist = r"D:\Projects\Major Project\SynapseRecon\simulated_images"
    orig_dist = os.path.join(base_dist, "original")
    scat_less_dist = os.path.join(base_dist, "scattered_less")
    
    os.makedirs(scat_less_dist, exist_ok=True)

    if not os.path.exists(orig_dist):
        print(f"Original directory not found: {orig_dist}")
        return

    image_files = [f for f in os.listdir(orig_dist) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {orig_dist}")

    for fname in image_files:
        input_path = os.path.join(orig_dist, fname)
        
        # Open image to get dimensions for grid calculation
        try:
            with Image.open(input_path) as img:
                w, h = img.size
                
            rows, cols = get_best_grid(w, h, 8, 12)
            print(f"Processing {fname} -> Grid: {rows}x{cols} ({rows*cols} pieces)")

            # Use simple filename in output
            out_scat = os.path.join(scat_less_dist, os.path.splitext(fname)[0] + ".jpg")
            scatter_image(input_path, rows=rows, cols=cols, output_path=out_scat)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    main()
