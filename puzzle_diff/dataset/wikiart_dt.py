from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# import tables as tb


class Wikiart_DT(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()
        import random
        # Seed for consistent split across runs
        random.seed(42)
        
        folder = Path("datasets/wikiart")
        # Try finding images recursively in case of subfolders
        all_images = sorted(list(folder.glob("**/*.jpg")) + list(folder.glob("**/*.png")))
        
        # Shuffle consistently
        random.shuffle(all_images)
        
        split_idx = int(len(all_images) * 0.8)
        
        if train:
            self.images = all_images[:split_idx]
        else:
            self.images = all_images[split_idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return Image.open(self.images[index]), None


# class Wikiart_DT_pytables(Dataset):
#   def __init__(self) -> None:
#       super().__init__()

#       # Open the existing HDF5 file
#       file = tb.open_file("wikiart_tr.h5", mode="r", cache_size=32 * 768 * 768 * 3)

#       self.data = file.root.images.images

#   def __len__(self):
#       return self.data.shape[0]

#   def __getitem__(self, index):
#       return Image.fromarray(self.data[index]), None
