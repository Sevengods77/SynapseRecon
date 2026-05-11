# Checkpoints

This folder holds the pretrained DiffAssemble model checkpoints used in Step 3 (Reassembly).

## Folder structure

```
checkpoints/
├── celeba/
│   └── <your_celeba_checkpoint>.ckpt
└── wikiart/
    └── <your_wikiart_checkpoint>.ckpt
```

## How to add checkpoints

1. Copy your `.ckpt` files into the correct sub-folder:
   - **CelebA model** → `checkpoints/celeba/`
   - **WikiArt model** → `checkpoints/wikiart/`

2. The filename doesn't matter — `api.py` auto-detects any `.ckpt` file present in each folder.

## Why checkpoints are not in Git

`.ckpt` files are typically hundreds of MB and are excluded via `.gitignore`.
Download them from the original [DiffAssemble](https://github.com/sepidmnoroozi/DiffAssemble) release or from the shared model storage for this project.
