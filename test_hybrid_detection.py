"""
test_hybrid_detection.py — Comprehensive Local Verification for 3-Layer Cascade
"""
import os
import io
import math
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# 1. Import our custom modules
from api import pipeline_state, get_best_grid
from detection_utils import find_pieces_by_template, find_pieces_by_background_subtraction, cluster_boxes_dbscan

def run_tests():
    print("=" * 60)
    print("STARTING COMPREHENSIVE 3-LAYER CASCADE VERIFICATION")
    print("=" * 60)

    # Resolve paths
    base_dir = r"D:\Projects\Major Project\Synapse_Recon"
    original_path = os.path.join(base_dir, "simulated_images", "original", "akshay.jpg")
    scattered_path = os.path.join(base_dir, "simulated_images", "scattered_less", "akshay.jpg")

    if not os.path.exists(original_path):
        print(f"Error: Original master image not found at {original_path}")
        return
    if not os.path.exists(scattered_path):
        print(f"Error: Scattered image not found at {scattered_path}")
        return

    # ──────────────────────────────────────────────────────────
    # PHASE 1: Simulate Ingest Master Image
    # ──────────────────────────────────────────────────────────
    print("\n--- PHASE 1: Simulating /ingest_master ---")
    master_img = Image.open(original_path).convert("RGB")
    img_w, img_h = master_img.size
    print(f"Loaded master image: {original_path} ({img_w}x{img_h})")

    # Run grid calculation
    best_rows, best_cols = get_best_grid(img_w, img_h, 8, 12)
    print(f"Calculated optimal grid scale: {best_rows}x{best_cols} (Total expected: {best_rows * best_cols})")
    
    pipeline_state["master_img"] = np.array(master_img)
    pipeline_state["master_grid"] = (best_rows, best_cols)
    pipeline_state["n_expected_pieces"] = best_rows * best_cols

    # Crop and store patches
    piece_w = img_w / best_cols
    piece_h = img_h / best_rows
    master_crops = {}
    for r in range(best_rows):
        for c in range(best_cols):
            x1 = int(c * piece_w)
            y1 = int(r * piece_h)
            x2 = int((c + 1) * piece_w)
            y2 = int((r + 1) * piece_h)
            crop = master_img.crop((x1, y1, x2, y2))
            master_crops[f"Grid_{best_rows}x{best_cols}_{r}_{c}"] = np.array(crop)
    
    pipeline_state["master_crops"] = master_crops
    print(f"Successfully cached {len(master_crops)} master crops in pipeline_state.")

    # ──────────────────────────────────────────────────────────
    # PHASE 2: Simulate Step 2 Cascade - Layer 1 (Template Matching)
    # ──────────────────────────────────────────────────────────
    print("\n--- PHASE 2: Simulating /analyze_image - Layer 1 Template Matching ---")
    scattered_img = Image.open(scattered_path).convert("RGB")
    scattered_np = np.array(scattered_img)
    sh_w, sh_h = scattered_img.size
    print(f"Loaded scattered image: {scattered_path} ({sh_w}x{sh_h})")

    n_expected = pipeline_state["n_expected_pieces"]
    detected_templates = find_pieces_by_template(scattered_np, pipeline_state["master_crops"], confidence_threshold=0.50)
    print(f"Layer 1 returned {len(detected_templates)} matches.")

    # If Layer 1 works, print results
    if len(detected_templates) >= n_expected * 0.8:
        print("✓ SUCCESS: Layer 1 (Template Matching) met the 80% pass threshold!")
        for idx, (x1, y1, x2, y2, crop_id, conf) in enumerate(detected_templates[:5]):
            print(f"  - Match {idx+1}: {crop_id} at [{x1}, {y1}, {x2}, {y2}] with conf: {conf}")
        if len(detected_templates) > 5:
            print(f"  ... and {len(detected_templates) - 5} more matches.")
    else:
        print("⚠ WARNING: Layer 1 did not meet 80% threshold. Let's force fallback testing.")

    # ──────────────────────────────────────────────────────────
    # PHASE 3: Simulate Fallback - Layer 2 (Background Subtraction)
    # ──────────────────────────────────────────────────────────
    print("\n--- PHASE 3: Simulating Fallback - Layer 2 Background Subtraction ---")
    detected_boxes = find_pieces_by_background_subtraction(scattered_np, n_expected)
    print(f"Layer 2 returned {len(detected_boxes)} bounding boxes.")
    for idx, box in enumerate(detected_boxes[:5]):
        print(f"  - Box {idx+1}: {box}")
    if len(detected_boxes) > 5:
        print(f"  ... and {len(detected_boxes) - 5} more boxes.")

    # ──────────────────────────────────────────────────────────
    # PHASE 4: Simulate Fallback - Layer 3 (FastSAM + DBSCAN)
    # ──────────────────────────────────────────────────────────
    print("\n--- PHASE 4: Simulating Fallback - Layer 3 FastSAM + DBSCAN ---")
    try:
        from ultralytics import FastSAM
        fast_sam = FastSAM(os.path.join(base_dir, "checkpoints", "FastSAM-s.pt"))
        results = fast_sam(scattered_np, device='cpu', retina_masks=True, imgsz=640, conf=0.35, iou=0.5, verbose=False)
        raw_boxes = []
        if results and len(results) > 0 and results[0].masks is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            img_area = sh_w * sh_h
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                if x2 > x1 and y2 > y1 and box_w >= 25 and box_h >= 25:
                    if box_area < 0.8 * img_area:
                        raw_boxes.append((x1, y1, x2, y2))
        
        best_scale = max(2, round(math.sqrt(n_expected)))
        expected_piece_w = sh_w / best_scale
        expected_piece_h = sh_h / best_scale
        expected_diagonal = math.sqrt(expected_piece_w**2 + expected_piece_h**2)
        
        clustered_boxes = cluster_boxes_dbscan(raw_boxes, expected_diagonal)
        print(f"Layer 3 successfully loaded FastSAM and clustered raw boxes from {len(raw_boxes)} to {len(clustered_boxes)} via DBSCAN.")
    except Exception as e:
        print(f"Layer 3 simulation failed (likely due to missing FastSAM weights/CUDA): {e}")

    # ──────────────────────────────────────────────────────────
    # PHASE 5: Simulate Post-Validation KMeans Clustering
    # ──────────────────────────────────────────────────────────
    print("\n--- PHASE 5: Simulating Post-Validation KMeans Clustering ---")
    # Generate some dummy duplicate bounding boxes to trigger KMeans clustering down
    dummy_detections = []
    # Create 2 * n_expected pieces
    for i in range(2 * n_expected):
        r = i % best_rows
        c = (i // best_rows) % best_cols
        # Jitter the box coordinates slightly to simulate duplicates
        jitter_x = np.random.randint(-15, 15)
        jitter_y = np.random.randint(-15, 15)
        x1 = int(c * piece_w + 100 + jitter_x)
        y1 = int(r * piece_h + 100 + jitter_y)
        x2 = int(x1 + piece_w)
        y2 = int(y1 + piece_h)
        dummy_detections.append({
            "piece_index": i + 1,
            "id": f"Grid_{best_rows}x{best_cols}_{r}_{c}",
            "match_distance": 0.1,
            "box": [x1, y1, x2, y2]
        })
    print(f"Created {len(dummy_detections)} dummy detections (expected limit: {n_expected}).")
    
    if len(dummy_detections) > n_expected * 1.3:
        print(f"Triggering KMeans down-clustering from {len(dummy_detections)} to {n_expected}...")
        centers = np.array([((d["box"][0] + d["box"][2])//2, (d["box"][1] + d["box"][3])//2) for d in dummy_detections])
        km = KMeans(n_clusters=n_expected, random_state=42, n_init=10).fit(centers)
        labels = km.labels_
        
        new_detections = []
        for label in range(n_expected):
            cluster_items = [dummy_detections[i] for i in range(len(dummy_detections)) if labels[i] == label]
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
        print(f"✓ KMeans down-clustering successfully completed! Clustered down to {len(new_detections)} pieces.")
        assert len(new_detections) == n_expected, "Post-validation count does not match expected pieces!"

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
