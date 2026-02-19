import os
from PIL import Image
from torch.utils.data import Dataset

class BreakHisDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []

        self._load_samples()

    def _load_samples(self):
        split_dir = os.path.join(self.root, self.split)
        classes = sorted(os.listdir(split_dir))

        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))

    
    def __len__(self):
        return len(self.samples)
    
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
