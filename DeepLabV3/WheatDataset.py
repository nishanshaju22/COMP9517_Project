from PIL import Image
import torch
import numpy as np
from DeepLabV3.transformers import get_image_mask_pairs

class WheatDataset(torch.utils.data.Dataset):

    def __init__(self, folder, transform):

        self.images, self.masks = get_image_mask_pairs(folder)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = np.array(
            Image.open(self.images[idx]).convert("RGB")
        )

        mask = np.array(
            Image.open(self.masks[idx]).convert("L")
        )

        mask = (mask > 127).astype(np.uint8)

        if self.transform:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0).float()

        return image, mask