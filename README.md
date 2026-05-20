<div align="center">

<br/>

```
███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗    ██████╗ ███████╗ ██████╗  ██████╗ ███╗   ██╗
██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝    ██╔══██╗██╔════╝██╔════╝ ██╔═══██╗████╗  ██║
███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗      ██████╔╝█████╗  ██║      ██║   ██║██╔██╗ ██║
╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝      ██╔══██╗██╔══╝  ██║      ██║   ██║██║╚██╗██║
███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗    ██║  ██║███████╗╚██████╗ ╚██████╔╝██║ ╚████║
╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
```

### *AI-Driven Puzzle Reconstruction Engine with 3-Layer Visual Processing & AR-Guided Assembly*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge&logo=vector)](https://trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-Google_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

[![GitHub stars](https://img.shields.io/github/stars/Sevengods77/SynapseRecon?style=social)](https://github.com/Sevengods77/SynapseRecon/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Sevengods77/SynapseRecon?style=social)](https://github.com/Sevengods77/SynapseRecon/network/members)
[![Last Commit](https://img.shields.io/github/last-commit/Sevengods77/SynapseRecon/Synapse_recon)](https://github.com/Sevengods77/SynapseRecon/commits/Synapse_recon)
[![Repo Size](https://img.shields.io/github/repo-size/Sevengods77/SynapseRecon)](https://github.com/Sevengods77/SynapseRecon)

</div>

---

<br/>

## 🧩 Overview

**SynapseRecon** is an advanced **2D puzzle reassembly pipeline** that leverages state-of-the-art Computer Vision, High-Dimensional Vector Search, Bipartite Optimization, and Generative Multimodal AI to guide users through the physical reconstruction of scattered puzzle pieces. 

By utilizing a standard camera feed, the system automatically detects, matches, and positions scattered puzzle fragments relative to their original coordinates, generating clear, real-time voice instructions and interactive Augmented Reality (AR) overlays.

---

## ⚙️ Architecture & Core Mechanics

The pipeline is split into **5 distinct phases**, moving from ingestion to interactive physical assembly guidance.

```
+-------------------------------------------------------------------------------------------------+
|                                     SYNAPSERCON PIPELINE FLOW                                   |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|  [Step 1: Master Ingestion] ────▶ [Step 2: Scattered Fragment Analysis] ───▶ [Step 3: Assembly] |
|            │                                      │                                 │           |
|            ▼                                      ▼                                 ▼           |
|    - Aspect Ratio Grid (8-12)             - 3-Layer Detection (NCC,          - Coordinate       |
|    - Multi-Scale Crop Pyramid               Contour, FastSAM+DBSCAN)           Synthesis        |
|    - DINOv2 Visual Features               - Bipartite (1-to-1) Matching      - Scattered-to-    |
|    - ChromaDB Embeddings                  - KMeans Box Validation              Target Mapping   |
|                                                                                     │           |
|                                                                                     ▼           |
|  [Step 5: AR Ghost Overlay] ◀─── [Step 4: Instruction Generation] ◄─────────────────┘           |
|            │                                      │                                             |
|            ▼                                      ▼                                             |
|    - WebRTC Camera Feed                   - Gemini 2.5 Flash VLM                                |
|    - Canvas Overlay                       - Contextual Instructions                             |
|    - Vocal/Speech Guidance                - JSON & Read File Saves                              |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

<br/>

### 📁 Core Module Files
- `api.py`: FastAPI server exposing the endpoints for all steps and handling pipeline state.
- `detection_utils.py`: Modular 3-layer image segmentation and clustering algorithms.
- `generate_fragments.py` & `generate_fragments_limited.py`: Puzzle piece scattering simulator for generating synthetic validation sets.
- `test_hybrid_detection.py`: Verification script to evaluate detection success across simulated image sets.
- `frontend/index.html`: Fully responsive vanilla frontend providing step navigation, real-time overlay visualization, and camera integrations.

---

## 🛠️ Step-by-Step Technical Deep-Dive

### 📂 Step 1: Ingest Master Image
When a reference/master image is uploaded, the backend prepares it for multi-scale matching:
1. **Adaptive Grid Calculation**: Computes the optimal grid structure (rows × cols) yielding a piece count between **8 and 12** while staying mathematically closest to the uploaded image's aspect ratio.
2. **Primary Crop Templates**: Generates crops corresponding to this grid size for template matching.
3. **Multi-Scale Feature Pyramid**: Builds an embedding index using grids ranging from $2\times2$ to $12\times12$.
4. **DINOv2 Embeddings**: For each crop at every scale, it runs the crop through `facebook/dinov2-small` to extract a high-dimensional feature representation.
5. **Vector Database Storage**: Upserts the generated vectors and coordinate metadatas into a local, persistent **ChromaDB** collection (`vector_db/`), flushing any stale previous indices.

```
                    +--------------------+
                    | Upload Master Img  |
                    +---------+----------+
                              |
                     [Aspect Ratio Check]
                              |
                    +---------▼----------+
                    | Optimal Primary    |
                    | Grid (e.g. 3x4)    |
                    +---------+----------+
                              |
               +--------------+--------------+
               |                             |
     [Primary Templates]           [Multi-Scale Pyramid]
     - Row/Col crop maps           - Scales 2x2 to 12x12
                                   - DINOv2 Feature Extractor
                                   - ChromaDB Vector Store
```

---

### 🔍 Step 2: Analyze Scattered Fragments
When a photo of the scattered pieces on the workspace is uploaded, the backend initiates a **3-Layer Hybrid Detection System** designed to survive variable lighting and noisy tables:

```
               +--------------------------------------+
               |    Scattered Image Upload (Step 2)   |
               +----------------───┬───---------------+
                                   |
                   [Layer 1: Template Matching]
                   - Normalised Cross-Correlation (NCC)
                   - ≥ 80% expected matches?
                                   │
                         ├── YES ──┼── NO ──┐
                         │                  │
                         ▼                  ▼
                   [Apply Layer 1]   [Layer 2: Background Subtraction]
                                     - Color boundary median threshold
                                     - Morphological Close/Open
                                     - External contour area filter
                                            │
                                            ├── Found valid pieces? ──▶ [Apply Layer 2]
                                            │
                                            └── NO ──▶ [Layer 3: FastSAM + DBSCAN]
                                                       - FastSAM Segmentation (conf=0.35)
                                                       - DBSCAN spatial box merging
```

1. **Layer 1: Template Matching (Primary)**:
   - Uses sub-pixel normalized cross-correlation (`cv2.TM_CCOEFF_NORMED`) to match the primary master templates against the scattered canvas.
   - Ideal for structured scenes. If this yields $\ge 80\%$ of the expected pieces with high confidence ($\ge 0.50$), it bypasses heavier vision networks.
2. **Layer 2: Background Subtraction (First Fallback)**:
   - Samples border pixels to compute a robust color median of the assembly table.
   - Computes absolute color distance, thresholding pixels to segment foreground puzzle pieces.
   - Applies morphological closing (to bridge inner gaps) and opening (to eliminate background speckle).
   - Isolates bounding boxes from external contours filtering by target surface area.
3. **Layer 3: FastSAM + DBSCAN Clustering (Last Resort)**:
   - Runs **FastSAM** (`FastSAM-s.pt`) to extract instance masks.
   - Applies **DBSCAN** spatial clustering on coordinate bounding box centers. If a single physical piece was segmented into multiple disjoint boxes, DBSCAN groups and merges them back into unified fragment boundaries.

#### ⚖️ Bipartite Matching Optimization
Once the physical coordinates are resolved, features are extracted for each detected piece using DINOv2 and queried against ChromaDB. 

To prevent the classic nearest-neighbor failure mode (where multiple physically distinct pieces claim the same target location), the system solves the **Linear Sum Assignment (Hungarian Algorithm)** using a cost matrix built from vector distances:

$$\min \sum_{i=1}^{N} \sum_{j=1}^{M} C_{i, j} X_{i, j}$$

$$\text{Subject to } \sum_{i} X_{i,j} = 1 \text{ and } \sum_{j} X_{i,j} = 1$$

If the system segments more fragments than physically expected, it executes a **KMeans Clustering** pass on the physical centers to aggregate coordinate bounding boxes down to the target piece count.

---

### 🗺️ Step 3: Reassembly Coordinate Mapping
Step 3 unifies the spatial data by mapping the coordinates of detected scattered fragments to their final target positions:
- Captures bounding box centers ($C_x, C_y$) of fragments on the scattered board.
- Resolves target coordinates ($T_x, T_y, T_w, T_h$) and scales using metadata retrieved from the DINOv2 vector lookup.
- Formulates localized, descriptive placement parameters (e.g. *"Currently sitting in the top-left area of the workspace. Needs to be placed in row 2, column 3"*).
- Outputs a comprehensive JSON response linking each piece's initial bounding box to its matching target bounding box in the reconstructed workspace.

---

### 🗣️ Step 4: Generate Instructions
Using Google's **Gemini 2.5 Flash Lite VLM**, Step 4 translates raw coordinates and direction parameters into intuitive, step-by-step assembly instructions.

```
       +---------------------------------------------------------------+
       | Step 3 Spatial Coordinate Output                              |
       +───────────────────────────────┬───────────────────────────────+
                                       |
                                       ▼
       +---------------------------------------------------------------+
       | VLM Generation Prompt & Constraints                           |
       | - Exactly one clear action instruction per fragment           |
       | - Translates indexes to regions ("top-left", "lower edge")    |
       | - Avoids repetitive transitions ("Next", "Then", "Now")       |
       | - No raw coordinates or column/row index verbal output        |
       +───────────────────────────────┬───────────────────────────────+
                                       |
                                       ▼
       +---------------------------------------------------------------+
       | API Responses & Local File Exports                            |
       | - JSON Payload to UI client                                   |
       | - `last_instructions.json` structured step log                |
       | - `assembly_instructions.txt` user-readable text file         |
       +---------------------------------------------------------------+
```

- **Output validation**: Standardizes instruction structures to compile directly into JSON arrays.
- **Exporting files**: Automatically writes instructions to `last_instructions.json` and creates a formatted human-readable guide in `assembly_instructions.txt`.

---

### 📷 Step 5: AR Ghost Overlay
The frontend consumes the unified data mapping to render real-time interactive cues:
1. **WebRTC Camera Stream**: Mounts the active camera feed directly to a responsive viewport container.
2. **Dynamic Canvas Context**: Superimposes transparent ghost silhouettes representing target locations, scaled and transformed over the corresponding video elements.
3. **Voice Guidance System**: Leverages the browser Web Speech API to read instructions dynamically as the user progresses. Includes navigation arrows to step forward or backward, alongside a dedicated vocal stop controller.

---

## 📊 Simulated Datasets & Test Statistics

To validate the reliability of the 3-Layer Hybrid Detection pipeline and the spatial assignments, a simulation suite is included in this repository under `simulated_images/`:

| Dataset Source | Original Images | High-Scatter Scenarios (`scattered/`) | Low-Scatter Scenarios (`scattered_less/`) | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **CelebA Subset** | 12 | ✅ (8-12 pieces) | ✅ (8-12 pieces) | Facial structures & continuous gradients |
| **WikiArt Subset** | 10 | ✅ (8-12 pieces) | ✅ (8-12 pieces) | Artistic textures, painting contours & landscapes |
| **Custom Real Images** | 3 | ✅ (8-12 pieces) | ✅ (8-12 pieces) | Varied shapes, high-frequency physical boundaries |

### 📈 Execution Performance Metrics

*Benchmarked on local execution runs using PyTorch CPU/CUDA fallback.*

- **Layer 1 Template Matching (NCC)**: `< 0.3s` (Rapid local array operations)
- **Layer 2 Background Contour Extraction**: `< 0.15s`
- **Layer 3 FastSAM Inference (imgsz=640)**: `~0.8s` (GPU) / `~2.5s` (CPU)
- **ChromaDB Spatial Query**: `< 0.05s`
- **Bipartite Linear Sum Assignment ($N=12$)**: `< 0.005s`
- **Gemini 2.5 Flash Instruction Generation**: `~1.8s` - `~3.0s`

---

## 🚀 Quick Start & Deployment

### 📋 Prerequisites
- **Python 3.10+**
- **Google Gemini API Key** (Set as environment variable)

### 📦 1. Installation
Clone the repository and install all required framework packages:
```bash
git clone https://github.com/Sevengods77/SynapseRecon.git
cd SynapseRecon
git checkout Synapse_recon

pip install -r requirements.txt
```

### 🔑 2. Environment Setup
Create a `.env` file in the root workspace directory:
```env
GEMINI_API_KEY=AIzaSy...your_gemini_api_key...
```

### 💾 3. Model Checkpoints
Verify that the segmentation model is present inside the `checkpoints/` directory:
- `checkpoints/FastSAM-s.pt` (Required for Layer 3 fallback execution)

### 🌐 4. Run the Pipeline Backend
Initiate the FastAPI server using Uvicorn:
```bash
python api.py
```
*The service will start listening on [http://0.0.0.0:8000](http://0.0.0.0:8000).*

### 🖥️ 5. Run the Interactive Frontend
Open `frontend/index.html` in any modern web browser to access the step-by-step UI dashboard. Ensure camera access is allowed for Step 5 AR functionality.

---

## 🛠️ Validation Testing
To run the automated validation tests and evaluate the hybrid detection system across the simulation dataset:
```bash
python test_hybrid_detection.py
```
This runs the detector across `simulated_images/`, displaying the detected pieces, target bounds, and matching statistics.

---

<div align="center">

**Developed with 🧠 FastAPI · ⚡ DINOv2 & ChromaDB · 🔮 Gemini AI**

*SynapseRecon — Reconstruction perfected through spatial intelligence.*

</div>
