import importlib.metadata as meta

pkgs = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "transformers": "transformers",
    "Pillow": "Pillow",
    "chromadb": "chromadb",
    "numpy": "numpy",
}

print(f"{'Package':<15} {'Status':<15} {'Version'}")
print("-" * 45)
for display, pkg in pkgs.items():
    try:
        version = meta.version(pkg)
        print(f"{display:<15} {'INSTALLED':<15} {version}")
    except meta.PackageNotFoundError:
        print(f"{display:<15} {'NOT FOUND':<15} -")

# Also check CUDA availability for torch
try:
    import torch
    cuda = f"CUDA available: {torch.cuda.is_available()}"
    if torch.cuda.is_available():
        cuda += f", version: {torch.version.cuda}"
    print(f"\n{cuda}")
except Exception:
    pass
