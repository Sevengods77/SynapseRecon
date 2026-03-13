from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# import tables as tb


class CelebA_HQ(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()
        if train:
            folder = Path("datasets/CelebA-HQ_train")
        else:
            folder = Path("datasets/CelebA-HQ_test")
        
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder {folder} not found.")

        self.images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))

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
