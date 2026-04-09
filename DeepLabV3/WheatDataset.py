from PIL import Image
import torch
from DeepLabV3.transformers import get_image_mask_pairs, mask_transform

class WheatDataset(torch.utils.data.Dataset):

    def __init__(self, folder, transform):

        self.images, self.masks = get_image_mask_pairs(folder)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(
            self.images[idx]
        ).convert("RGB")

        mask = Image.open(
            self.masks[idx]
        ).convert("L")

        image = self.transform(image)

        mask = mask_transform(mask)

        return image, mask