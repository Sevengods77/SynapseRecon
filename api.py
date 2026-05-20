
import sys
import os
import math
import numpy as np
import base64
import torch
import chromadb
import einops
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from transformers import AutoImageProcessor, AutoModel
from ultralytics import FastSAM
from PIL import Image
import io
import json
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import torchvision.transforms as transforms
import torch_geometric as pyg
import torch_geometric.data as pyg_data
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from detection_utils import find_pieces_by_template, find_pieces_by_background_subtraction, cluster_boxes_dbscan

# Load environment variables (API Keys)
load_dotenv()

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


# Add puzzle_diff to Python path so DiffAssemble model can be imported
# Uses a path relative to this file — works on any machine after cloning the repo
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE_DIR, "puzzle_diff"))

try:
    from model import spatial_diffusion_on_angle as sd
    DIFFASSEMBLE_AVAILABLE = True
    print("DiffAssemble module found.")
except ImportError as e:
    DIFFASSEMBLE_AVAILABLE = False
    print(f"Warning: DiffAssemble not available: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
SYSTEM_INSTRUCTION = """
You are a friendly, natural-sounding puzzle assembly assistant guiding a beginner. Convert the provided spatial data into clear, conversational voice instructions.
Rules:
1. Exactly one instruction per fragment.
2. NEVER use the word "next". Use varied transitions like "Then", "Now", "Following that", "Moving on".
3. DO NOT output robotic coordinates or say "row X", "column Y", or "out of". Instead, mentally translate the grid coordinates into natural regions of the puzzle (e.g., "the top-left corner", "the upper edge", "the lower-right area", "the center").
4. Identify pieces using descriptive labels based on their starting area (e.g., "Find the fragment in the bottom right of your workspace", "Look for the piece on the top left"), rather than just calling them "Piece 1".
5. Describe the movement naturally (e.g., "Slide it upwards and slightly to the right to form the top edge").
6. Output ONLY a raw JSON array of strings, without any markdown formatting or introductory text.
"""
vlm_model = genai.GenerativeModel(
    'gemini-2.5-flash-lite',
    system_instruction=SYSTEM_INSTRUCTION,
    generation_config={
        "temperature": 0.1,
        "max_output_tokens": 1000,
        "response_mime_type": "application/json"
    }
)

class Phase3Data(BaseModel):
    fragment_data: list

# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
print("Loading FastSAM model...")
fast_sam = FastSAM(os.path.join(_BASE_DIR, "checkpoints", "FastSAM-s.pt"))

print("Loading DINOv2 model (HuggingFace)...")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
dino_model = AutoModel.from_pretrained('facebook/dinov2-small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_model.to(device)
dino_model.eval()

# Connect to ChromaDB
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection(name="master_features")

# Load DiffAssemble Checkpoints (files live inside this repo under checkpoints/)
CHECKPOINTS = {
    "celeba":  os.path.join(_BASE_DIR, "checkpoints", "celeba",  "epoch=111-overall_acc=0.7507-overall__piece_acc=0.9692.ckpt"),
    "wikiart": os.path.join(_BASE_DIR, "checkpoints", "wikiart", "epoch=44-overall_acc=0.7236-overall__piece_acc=0.9599.ckpt"),
}
diff_models = {}
_img_transforms = transforms.Compose([transforms.ToTensor()])

if DIFFASSEMBLE_AVAILABLE:
    # Temporarily patch torch.load to bypass weights_only=True security error
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load

    for mode, path in CHECKPOINTS.items():
        if os.path.exists(path):
            try:
                diff_models[mode] = sd.GNN_Diffusion.load_from_checkpoint(path).to(device)
                diff_models[mode].eval()
                diff_models[mode].noise_weight = 1
                print(f"✓ DiffAssemble [{mode}] loaded.")
            except Exception as e:
                print(f"FAILED to load DiffAssemble [{mode}]: {e}")
        else:
            print(f"Warning: Checkpoint not found — {path}")
            
    # Restore original torch.load
    torch.load = _orig_load

# ─────────────────────────────────────────────
# In-memory pipeline state (shared between Step 2 → Step 3)
# ─────────────────────────────────────────────
pipeline_state = {
    "image_bytes": None,       # raw bytes of the scattered image
    "detections": [],          # list of {piece_index, matched_grid_id, match_distance, current_box}
    "master_crops": {},        # dict of { "Grid_RxC_r_c": np_array, ... } for template matching
    "master_img": None,        # numpy array of the master image
    "n_expected_pieces": 9,    # expected piece count
    "master_grid": (3, 3)      # (rows, cols) of the master grid
}

print("Backend initialized and ready.")


# ─────────────────────────────────────────────
# STEP 1: Ingest Master Image
# ─────────────────────────────────────────────
@app.post("/ingest_master")
async def ingest_master(file: UploadFile = File(...)):
    print(f"[Step 1] Received master image: {file.filename}")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img_width, img_height = image.size

        # Save master image to pipeline_state
        pipeline_state["master_img"] = np.array(image)

        # Compute best grid size (primary scale)
        best_rows, best_cols = get_best_grid(img_width, img_height, 8, 12)
        pipeline_state["master_grid"] = (best_rows, best_cols)
        pipeline_state["n_expected_pieces"] = best_rows * best_cols

        # Crop primary scale master crops for template matching
        piece_w = img_width / best_cols
        piece_h = img_height / best_rows
        master_crops = {}
        for r in range(best_rows):
            for c in range(best_cols):
                x1 = int(c * piece_w)
                y1 = int(r * piece_h)
                x2 = int((c + 1) * piece_w)
                y2 = int((r + 1) * piece_h)

                crop = image.crop((x1, y1, x2, y2))
                master_crops[f"Grid_{best_rows}x{best_cols}_{r}_{c}"] = np.array(crop)
        pipeline_state["master_crops"] = master_crops
        print(f"  Saved master crops for {best_rows}x{best_cols} grid in pipeline_state.")

        # Build a multi-scale spatial pyramid — supports any number of puzzle pieces
        grid_scales = [2, 3, 4, 5, 6, 8, 10, 12]

        ingested_count = 0

        # Clear existing data
        existing_data = collection.get()
        if existing_data and existing_data.get('ids'):
            collection.delete(ids=existing_data['ids'])
            print(f"  Cleared {len(existing_data['ids'])} existing entries from database.")

        for scale in grid_scales:
            piece_w = img_width / scale
            piece_h = img_height / scale

            for r in range(scale):
                for c in range(scale):
                    x1 = int(c * piece_w)
                    y1 = int(r * piece_h)
                    x2 = int((c + 1) * piece_w)
                    y2 = int((r + 1) * piece_h)

                    crop = image.crop((x1, y1, x2, y2))

                    inputs = processor(images=crop, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = dino_model(**inputs)

                    features = outputs.pooler_output.squeeze().cpu().numpy().tolist()
                    piece_id = f"Grid_{scale}x{scale}_{r}_{c}"

                    collection.add(
                        embeddings=[features],
                        metadatas=[{"box": f"{x1},{y1},{x2},{y2}", "scale": str(scale), "row": str(r), "col": str(c)}],
                        ids=[piece_id]
                    )
                    ingested_count += 1

        return {
            "success": True,
            "ingested_count": ingested_count,
            "message": f"Successfully ingested {ingested_count} pieces across multiple scales."
        }
    except Exception as e:
        print(f"[Step 1] Ingestion error: {e}")
        return {"success": False, "error": str(e), "ingested_count": 0}


# ─────────────────────────────────────────────
# STEP 2: Analyze Scattered Fragments
# ─────────────────────────────────────────────
@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    print(f"[Step 2] Received scattered image: {file.filename}")
    try:
        contents = await file.read()

        # Save raw bytes into pipeline_state for Step 3
        pipeline_state["image_bytes"] = contents
        pipeline_state["detections"] = []

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(image)
        img_w, img_h = image.size

        n_expected = pipeline_state.get("n_expected_pieces", 9)
        use_template_results = False
        final_detections = []

        # Layer 1: Template Matching (primary)
        master_crops = pipeline_state.get("master_crops")
        if master_crops:
            print(f"[Layer 1] Attempting Template Matching for {n_expected} expected pieces...")
            # Try to match crops in frame using find_pieces_by_template
            detected_templates = find_pieces_by_template(frame, master_crops, confidence_threshold=0.50)
            print(f"[Layer 1] Found {len(detected_templates)} matches.")
            
            # If we detected at least 80% of expected pieces, we use Layer 1
            if len(detected_templates) >= n_expected * 0.8:
                print(f"[Layer 1] High confidence match count ({len(detected_templates)}/{n_expected}). Proceeding with Layer 1.")
                use_template_results = True
                
                # Format detection data
                # detected_templates elements are: (x1, y1, x2, y2, crop_id, confidence)
                for idx, (x1, y1, x2, y2, crop_id, conf) in enumerate(detected_templates):
                    piece_data = {
                        "piece_index": idx + 1,
                        "id": crop_id,
                        "match_distance": round(1.0 - conf, 4), # match_distance = 1 - confidence
                        "box": [x1, y1, x2, y2],
                    }
                    final_detections.append(piece_data)

        # Fallback to Layer 2 or Layer 3
        if not use_template_results:
            print(f"[Layer 2] Attempting Background Subtraction for {n_expected} expected pieces...")
            detected_boxes = find_pieces_by_background_subtraction(frame, n_expected)
            print(f"[Layer 2] Found {len(detected_boxes)} pieces.")
            
            # If Layer 2 fails to find at least 50% of expected pieces, fallback to Layer 3
            if len(detected_boxes) < n_expected * 0.5:
                print(f"[Layer 3] Background Subtraction found too few pieces. Falling back to Layer 3: FastSAM + DBSCAN...")
                
                # Run FastSAM
                results = fast_sam(frame, device='cpu', retina_masks=True, imgsz=640, conf=0.35, iou=0.5, verbose=False)
                raw_boxes = []
                if results and len(results) > 0 and results[0].masks is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    img_area = img_w * img_h
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_area = box_w * box_h
                        if x2 > x1 and y2 > y1 and box_w >= 25 and box_h >= 25:
                            if box_area < 0.8 * img_area:
                                raw_boxes.append((x1, y1, x2, y2))
                
                # Cluster boxes with DBSCAN
                best_scale = max(2, round(math.sqrt(n_expected)))
                expected_piece_w = img_w / best_scale
                expected_piece_h = img_h / best_scale
                expected_diagonal = math.sqrt(expected_piece_w**2 + expected_piece_h**2)
                
                detected_boxes = cluster_boxes_dbscan(raw_boxes, expected_diagonal)
                print(f"[Layer 3] DBSCAN clustered into {len(detected_boxes)} pieces.")
            
            # Now we have detected_boxes from Layer 2 or Layer 3. We extract features and match them.
            num_pieces = len(detected_boxes)
            best_scale = max(2, round(math.sqrt(num_pieces))) if num_pieces > 0 else 3
            print(f"  Detected {num_pieces} fragments. Forcing grid matching to {best_scale}x{best_scale}.")
            
            # Extract features for all pieces
            all_features = []
            piece_boxes = []
            for idx, box in enumerate(detected_boxes):
                x1, y1, x2, y2 = box
                crop = image.crop((x1, y1, x2, y2))
                inputs = processor(images=crop, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                features = outputs.pooler_output.squeeze().cpu().numpy().tolist()
                all_features.append(features)
                piece_boxes.append((x1, y1, x2, y2))
                
            # Perform optimal 1-to-1 Bipartite Matching (Linear Sum Assignment)
            matched_ids = ["Unknown"] * num_pieces
            match_distances = [None] * num_pieces
            
            if num_pieces > 0:
                try:
                    from scipy.optimize import linear_sum_assignment
                    db_results = collection.query(
                        query_embeddings=all_features,
                        n_results=best_scale * best_scale, # Fetch all grid cells for this scale
                        where={"scale": str(best_scale)}
                    )
                    
                    # Collect all unique cell IDs
                    unique_cell_ids = []
                    for ids_list in db_results['ids']:
                        for cid in ids_list:
                            if cid not in unique_cell_ids:
                                unique_cell_ids.append(cid)
                                
                    # Build Cost Matrix (rows = pieces, cols = unique grid cells)
                    cost_matrix = np.full((num_pieces, len(unique_cell_ids)), 1000.0)
                    for i in range(num_pieces):
                        dists = db_results['distances'][i]
                        ids = db_results['ids'][i]
                        for j, cid in enumerate(ids):
                            col_idx = unique_cell_ids.index(cid)
                            cost_matrix[i, col_idx] = dists[j]
                            
                    # Solve assignment to guarantee no two pieces are assigned to the same grid cell
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for i, r in enumerate(row_ind):
                        c = col_ind[i]
                        matched_ids[r] = unique_cell_ids[c]
                        match_distances[r] = round(cost_matrix[r, c], 4)
                except Exception as e:
                    print(f"Bipartite matching failed: {e}. Falling back to greedy assignment.")
                    used_ids = set()
                    for i in range(num_pieces):
                        try:
                            db_result = collection.query(
                                query_embeddings=[all_features[i]], 
                                n_results=best_scale * best_scale,
                                where={"scale": str(best_scale)}
                            )
                            for j, cid in enumerate(db_result['ids'][0]):
                                if cid not in used_ids:
                                    matched_ids[i] = cid
                                    match_distances[i] = round(db_result['distances'][0][j], 4)
                                    used_ids.add(cid)
                                    break
                        except Exception as qe:
                            matched_ids[i] = "Error"
                            
            for idx in range(num_pieces):
                x1, y1, x2, y2 = piece_boxes[idx]
                piece_data = {
                    "piece_index": idx + 1,
                    "id": matched_ids[idx],
                    "match_distance": match_distances[idx],
                    "box": [x1, y1, x2, y2],
                }
                final_detections.append(piece_data)

        # Post-validation / Clustering (If we got too many boxes, cluster down to n_expected using KMeans)
        if len(final_detections) > n_expected * 1.3:
            print(f"WARNING: Got {len(final_detections)} pieces, but expected {n_expected}. Clustering down to {n_expected} using KMeans...")
            try:
                centers = np.array([((d["box"][0] + d["box"][2])//2, (d["box"][1] + d["box"][3])//2) for d in final_detections])
                km = KMeans(n_clusters=n_expected, random_state=42, n_init=10).fit(centers)
                labels = km.labels_
                
                new_detections = []
                for label in range(n_expected):
                    cluster_items = [final_detections[i] for i in range(len(final_detections)) if labels[i] == label]
                    if not cluster_items:
                        continue
                    cx1 = min(item["box"][0] for item in cluster_items)
                    cy1 = min(item["box"][1] for item in cluster_items)
                    cx2 = max(item["box"][2] for item in cluster_items)
                    cy2 = max(item["box"][3] for item in cluster_items)
                    valid_items = [item for item in cluster_items if item["id"] != "Unknown" and item["id"] != "Error"]
                    best_item = min(valid_items, key=lambda x: x.get("match_distance", 1.0)) if valid_items else cluster_items[0]
                    
                    new_detections.append({
                        "piece_index": label + 1,
                        "id": best_item["id"],
                        "match_distance": best_item.get("match_distance"),
                        "box": [cx1, cy1, cx2, cy2]
                    })
                final_detections = new_detections
            except Exception as ke:
                print(f"KMeans clustering down failed: {ke}")

        # Update pipeline state detections
        response_data = []
        for idx, det in enumerate(final_detections):
            x1, y1, x2, y2 = det["box"]
            piece_data = {
                "piece_index": idx + 1,
                "id": det["id"],
                "match_distance": det["match_distance"],
                "box": [x1, y1, x2, y2],
            }
            response_data.append(piece_data)
            
            pipeline_state["detections"].append({
                "piece_index": idx + 1,
                "matched_grid_id": det["id"],
                "match_distance": det["match_distance"],
                "current_box": [x1, y1, x2, y2],
                "img_w": img_w,
                "img_h": img_h,
            })

        print(f"  Detected {len(response_data)} pieces. Pipeline state updated.")
        return {"detections": response_data}
    except Exception as e:
        print(f"[Step 2] Analysis error: {e}")
        return {"error": str(e), "detections": []}


# ─────────────────────────────────────────────
# STEP 3: Reassemble using DiffAssemble
# ─────────────────────────────────────────────
@app.post("/reassemble")
async def reassemble(mode: str = Form("celeba")):
    print(f"[Step 3] Reassembly requested with mode: {mode}")
    print(f"DEBUG: DIFFASSEMBLE_AVAILABLE = {DIFFASSEMBLE_AVAILABLE}")

    if not pipeline_state["image_bytes"]:
        return {"error": "No scattered image found. Please run Step 2 first."}

    if not pipeline_state["detections"]:
        return {"error": "No pieces detected from Step 2. Please analyze the scattered image first."}

    if not DIFFASSEMBLE_AVAILABLE:
        return {"error": "DiffAssemble module not available. Check puzzle_diff path."}

    if mode not in diff_models:
        return {"error": f"DiffAssemble model for '{mode}' not loaded. Check checkpoint path."}

    try:
        image = Image.open(io.BytesIO(pipeline_state["image_bytes"])).convert("RGB")
        detections = pipeline_state["detections"]
        
        # Get Master Image Dimensions from DB
        master_width, master_height = 800, 800 # defaults
        try:
            db_any = collection.get(limit=1)
            if db_any and db_any.get('metadatas') and len(db_any['metadatas']) > 0:
                meta = db_any['metadatas'][0]
                mx1, my1, mx2, my2 = map(float, meta['box'].split(','))
                scale = int(meta['scale'])
                master_width = int((mx2 - mx1) * scale)
                master_height = int((my2 - my1) * scale)
        except Exception as e:
            print("Could not infer master size from DB:", e)

        # DiffAssemble expects a fixed 6x6 grid of 32x32 patches
        patch_size = 32
        patch_dim_h, patch_dim_w = 6, 6
        patch_per_dim = [patch_dim_h, patch_dim_w]

        height = patch_dim_h * patch_size  # 192
        width = patch_dim_w * patch_size   # 192
        img_resized = image.resize((width, height), resample=Image.BICUBIC)
        img_tensor = _img_transforms(img_resized)

        xy, patches = _divide_into_patches(img_tensor, patch_per_dim, patch_size)
        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        n_patches = xy.shape[0]
        adj_mat = torch.ones(n_patches, n_patches)
        edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)

        graph_in = pyg_data.Data(
            x=xy,
            patches=patches,
            edge_index=edge_index,
            patches_dim=torch.tensor([patch_per_dim]),
            batch=torch.zeros(n_patches, dtype=torch.long)
        )

        diff_model = diff_models[mode]
        graph_in = graph_in.to(diff_model.device)

        with torch.no_grad():
            imgs, _ = diff_model.prediction_step(graph_in, 0)

        predicted_positions = imgs[-1]  # shape (36, 2) for 6x6

        # Map predicted normalized positions back to pixel coordinates scaled to master image
        diff_results = []
        for p in range(predicted_positions.shape[0]):
            x_norm = predicted_positions[p, 0].item()
            y_norm = predicted_positions[p, 1].item()
            da_tx = int((x_norm + 1) * master_width / 2)
            da_ty = int((y_norm + 1) * master_height / 2)
            diff_results.append({"patch_idx": p, "target_x": da_tx, "target_y": da_ty})

        # ─── Merge Step 2 detections with Step 3 DiffAssemble predictions ───
        unified_results = []
        for i, det in enumerate(detections):
            matched_id = det["matched_grid_id"]
            current_box = det["current_box"]
            img_w = det["img_w"]
            img_h = det["img_h"]

            # Current center of the detected piece in the scattered image
            cx = (current_box[0] + current_box[2]) / 2
            cy = (current_box[1] + current_box[3]) / 2

            # Parse Grid ID and fetch exact master coordinates from DB
            target_x, target_y, rotation_deg = None, None, 0
            target_w, target_h = None, None
            direction_hint = "Unknown"
            
            if matched_id != "Unknown":
                try:
                    db_res = collection.get(ids=[matched_id])
                    if db_res and db_res.get('metadatas') and len(db_res['metadatas']) > 0:
                        meta = db_res['metadatas'][0]
                        mx1, my1, mx2, my2 = map(float, meta['box'].split(','))
                        
                        target_x = int((mx1 + mx2) / 2)
                        target_y = int((my1 + my2) / 2)
                        target_w = int(mx2 - mx1)
                        target_h = int(my2 - my1)
                        
                        grid_row = int(meta['row'])
                        grid_col = int(meta['col'])
                        scale = int(meta['scale'])

                        # Human-readable direction hint for VLM (Phase 4)
                        start_h = "left" if cx < img_w / 3 else "right" if cx > 2 * img_w / 3 else "center"
                        start_v = "top" if cy < img_h / 3 else "bottom" if cy > 2 * img_h / 3 else "middle"
                        start_pos = "center" if start_h == "center" and start_v == "middle" else f"{start_v} {start_h}"
                        
                        direction_hint = f"Currently sitting in the {start_pos} area of the workspace. Needs to be placed in row {grid_row + 1} (out of {scale}) and column {grid_col + 1} (out of {scale}) of the final image."
                except Exception as e:
                    print(f"Error fetching exact target for {matched_id}: {e}")

            # Use DiffAssemble prediction if available for this patch index
            da = diff_results[i] if i < len(diff_results) else {}

            unified_results.append({
                "piece_index": det["piece_index"],
                "current_box": current_box,
                "current_center_x": int(cx),
                "current_center_y": int(cy),
                "matched_grid_id": matched_id,
                "match_confidence": round(1 - (det["match_distance"] or 1), 4),
                "target_x": target_x if target_x is not None else da.get("target_x"),
                "target_y": target_y if target_y is not None else da.get("target_y"),
                "diffassemble_target_x": da.get("target_x"),
                "diffassemble_target_y": da.get("target_y"),
                "target_w": target_w if target_w is not None else int(master_width / 6),
                "target_h": target_h if target_h is not None else int(master_height / 6),
                "rotation_deg": rotation_deg,
                "direction_hint": direction_hint,
            })

        print(f"  Unified {len(unified_results)} pieces for Phase 4.")
        
        # Determine actual grid scale used for drawing
        used_scale = 3
        if unified_results and unified_results[0].get("matched_grid_id", "Unknown") != "Unknown":
            try:
                used_scale = int(unified_results[0]["matched_grid_id"].split("_")[1].split("x")[0])
            except:
                pass
                
        return {
            "success": True,
            "mode_used": mode,
            "master_width": master_width,
            "master_height": master_height,
            "grid_scale": used_scale,
            "results": unified_results
        }

    except Exception as e:
        import traceback
        print(f"[Step 3] Reassembly error: {e}")
        traceback.print_exc()
        return {"error": str(e), "results": []}


# ─────────────────────────────────────────────
# STEP 4: Generate Instructions (VLM)
# ─────────────────────────────────────────────
@app.post("/generate_steps")
def generate_steps(data: Phase3Data):
    import time
    print(f"[Step 4] Generating instructions for {len(data.fragment_data)} pieces...")
    
    start_time = time.time()
    hints = [f"Piece {p.get('piece_index', '?')}: {p.get('direction_hint', 'matched location')}" for p in data.fragment_data]
    
    prompt = f"Spatial Data:\n{hints}"
    
    try:
        import requests
        import re
        api_key = os.environ.get("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1000,
                "responseMimeType": "application/json"
            }
        }
        resp = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload, timeout=15)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Robust JSON formatting extraction
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            steps = json.loads(match.group(0))
        else:
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            steps = json.loads(text.strip())
        
        if isinstance(steps, dict):
            for v in steps.values():
                if isinstance(v, list):
                    steps = v
                    break

        # Save to file for user verification (JSON)
        with open("last_instructions.json", "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "piece_count": len(data.fragment_data),
                "instructions": steps
            }, f, indent=4)
        
        # Save as a readable text file
        with open("assembly_instructions.txt", "w", encoding="utf-8") as f:
            f.write(f"Assembly Instructions Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Pieces: {len(data.fragment_data)}\n")
            f.write("-" * 50 + "\n")
            for idx, step in enumerate(steps):
                f.write(f"Piece {idx + 1}:\n  {step}\n\n")
                
        print(f"  Instructions saved to last_instructions.json and assembly_instructions.txt")

        duration = time.time() - start_time
        print(f"  VLM generation completed in {duration:.2f}s")
            
    except Exception as e:
        print(f"  VLM generation error: {e}")
        steps = [f"For piece {p.get('piece_index', '?')}, {p.get('direction_hint', 'its correct location')}." for p in data.fragment_data]
        
    return {"steps": steps}


def _divide_into_patches(img, patch_per_dim, patch_size):
    """Divide image tensor into non-overlapping patches and return grid positions."""
    img2 = img.permute(1, 2, 0)
    patches = img2.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    y = torch.linspace(-1, 1, patch_per_dim[0])
    x = torch.linspace(-1, 1, patch_per_dim[1])
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    return xy, patches


# ─────────────────────────────────────────────
# Legacy WebSocket (kept for compatibility)
# ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")
    try:
        while True:
            data = await websocket.receive_text()
            if not data.startswith("data:image"):
                continue
            image_data = base64.b64decode(data.split(",")[1])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            frame = np.array(image)
            results = fast_sam(frame, device='cpu', retina_masks=True, imgsz=640, conf=0.60, iou=0.30, verbose=False)
            response_data = []
            if results and len(results) > 0 and results[0].masks is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 25 or (y2 - y1) < 25: continue
                    crop = image.crop((x1, y1, x2, y2))
                    inputs = processor(images=crop, return_tensors="pt").to(device)
                    with torch.no_grad(): outputs = dino_model(**inputs)
                    features = outputs.pooler_output.squeeze().cpu().numpy().tolist()
                    try:
                        db_result = collection.query(query_embeddings=[features], n_results=1)
                        piece_id = db_result['ids'][0][0] if (db_result and db_result.get('ids') and len(db_result['ids'][0]) > 0) else "Unknown"
                    except: piece_id = "Error"
                    response_data.append({"id": piece_id, "box": [x1, y1, x2, y2]})
            await websocket.send_text(json.dumps(response_data))
    except Exception as e:
        print(f"WebSocket closed or error: {e}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
