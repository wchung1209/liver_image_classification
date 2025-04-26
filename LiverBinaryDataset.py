import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LiverBinaryDataset(Dataset):
    """
    PyTorch Dataset for binary classification of liver ultrasound images.

    Expects a pandas DataFrame with columns:
        - 'image': path to the JPEG file
        - 'liver': path to the liver JSON polygon
        - 'outline': path to the outline JSON polygon
        - 'mass': path to the mass JSON polygon (or None)
        - 'label': binary label (0 or 1)

    Returns (image_tensor, mask_tensor, label_tensor) per sample:
        - image_tensor: FloatTensor, shape (3, H, W), values in [0,1]
        - mask_tensor:  FloatTensor, shape (3, H, W), binary masks
        - label_tensor: FloatTensor, shape (), binary {0.,1.}
    """
    def __init__(self, df, img_transform=None):
        self.df = df.reset_index(drop=True)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) Load image (BGR -> RGB)
        img_bgr = cv2.imread(row['image'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # 2) Rasterize polygons into masks
        masks = []
        for key in ['liver', 'outline', 'mass']:
            mask = np.zeros((h, w), dtype=np.uint8)
            path = row[key]
            if path and os.path.exists(path):
                try:
                    coords = json.load(open(path))
                    if isinstance(coords, list) and len(coords) >= 3:
                        pts = np.array(coords, dtype=np.int32)
                        if pts.ndim == 2 and pts.shape[1] == 2:
                            pts = pts.reshape(-1, 1, 2)
                            cv2.fillPoly(mask, [pts], 1)
                except Exception:
                    pass
            masks.append(mask)
        mask_stack = np.stack(masks, axis=0)

        # 3) Apply optional transforms (only normalization, no augmentation)
        if self.img_transform:
            img_tensor = self.img_transform(torch.from_numpy(img_rgb)
                                           .permute(2,0,1).float().div(255.0))
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).float().div(255.0)

        mask_tensor = torch.from_numpy(mask_stack).float()
        label_tensor = torch.tensor(row['label'], dtype=torch.float32)

        return img_tensor, mask_tensor, label_tensor
