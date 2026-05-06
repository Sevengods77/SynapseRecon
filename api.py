import cv2
import numpy as np
import base64
import torch
import chromadb
from fastapi import FastAPI, WebSocket, UploadFile, File
from transformers import AutoImageProcessor, AutoModel
from ultralytics import FastSAM
from PIL import Image
import io
import json
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
print("Loading FastSAM model...")
fast_sam = FastSAM('FastSAM-s.pt') 

print("Loading DINOv2 model...")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Connect to DB
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection(name="master_features")
print("Backend initialized and ready.")

@app.post("/ingest_master")
async def ingest_master(file: UploadFile = File(...)):
    print(f"Received master image for ingestion: {file.filename}")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(image)
        
        img_width, img_height = image.size
        
        # Build a multi-scale spatial pyramid (automatically supports different piece sizes)
        grid_scales = [4, 6, 8, 12]
        
        ingested_count = 0
        
        # Clear existing data in the collection
        existing_data = collection.get()
        if existing_data and existing_data.get('ids'):
            collection.delete(ids=existing_data['ids'])
            print(f"Cleared {len(existing_data['ids'])} existing entries from database.")
            
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
                    
                    # DINOv2 Feature Extraction
                    inputs = processor(images=crop, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    features = outputs.pooler_output.squeeze().cpu().numpy().tolist()
                    piece_id = f"Grid_{scale}x{scale}_{r}_{c}"
                    
                    collection.add(
                        embeddings=[features],
                        metadatas=[{"box": f"{x1},{y1},{x2},{y2}", "type": "master_grid"}],
                        ids=[piece_id]
                    )
                    ingested_count += 1
                
        return {"success": True, "ingested_count": ingested_count, "message": f"Successfully ingested {ingested_count} pieces across multiple scales."}
    except Exception as e:
        print(f"Ingestion error: {e}")
        return {"success": False, "error": str(e), "ingested_count": 0}

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(image)
        
        # Run FastSAM
        results = fast_sam(frame, device='cpu', retina_masks=True, imgsz=640, conf=0.4, iou=0.9, verbose=False)
        
        response_data = []
        if results and len(results) > 0 and results[0].masks is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                # Prevent zero-area crops
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = image.crop((x1, y1, x2, y2))
                
                # DINOv2 Feature Extraction
                inputs = processor(images=crop, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                features = outputs.pooler_output.squeeze().cpu().numpy().tolist()
                
                # Query ChromaDB
                try:
                    db_result = collection.query(query_embeddings=[features], n_results=1)
                    if db_result and db_result.get('ids') and len(db_result['ids'][0]) > 0:
                        piece_id = db_result['ids'][0][0]
                    else:
                        piece_id = "Unknown"
                except Exception as qe:
                    print(f"Chroma query error: {qe}")
                    piece_id = "Error"
                
                response_data.append({
                    "id": piece_id,
                    "box": [x1, y1, x2, y2]
                })
        
        return {"detections": response_data}
    except Exception as e:
        print(f"Analysis error: {e}")
        return {"error": str(e), "detections": []}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Kept for legacy compatibility
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
            results = fast_sam(frame, device='cpu', retina_masks=True, imgsz=640, conf=0.4, iou=0.9, verbose=False)
            response_data = []
            if results and len(results) > 0 and results[0].masks is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    if x2 <= x1 or y2 <= y1: continue
                    crop = image.crop((x1, y1, x2, y2))
                    inputs = processor(images=crop, return_tensors="pt").to(device)
                    with torch.no_grad(): outputs = model(**inputs)
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
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
