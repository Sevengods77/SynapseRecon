import torch
import chromadb
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import uuid

# 1. Configuration
IMAGE_PATH = "master_images/akshay.jpg"
DB_PATH = "./vector_db"
COLLECTION_NAME = "master_features"

def main():
    # 2. Initialize ChromaDB (Local persistent storage)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Get or create a collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # 3. Load DINOv2 via Hugging Face Transformers
    print("Loading DINOv2 model...")
    # Using dinov2-small for laptop performance. Can upgrade to base/large if VRAM permits.
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set to evaluation mode (no training)
    print(f"Model loaded on {device}")

    # 4. Load and Process the Master Image
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Place your master image at {IMAGE_PATH}")
        
    print(f"Processing image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    # Prepare image for the model
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 5. Extract Features
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(**inputs)
        
        # We extract the 'pooler_output' which represents the global image features
        # For patch-level (piece-level) matching later, you can also extract last_hidden_state
        features = outputs.pooler_output.squeeze().cpu().numpy()

    # 6. Store in Vector Database
    image_id = str(uuid.uuid4())
    
    collection.add(
        embeddings=[features.tolist()],
        metadatas=[{"filename": "original_artwork.jpg", "type": "master_reference"}],
        ids=[image_id]
    )
    
    print(f"Success! Extracted vector of size {features.shape}.")
    print(f"Features saved to ChromaDB with ID: {image_id}")

if __name__ == "__main__":
    main()