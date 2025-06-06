def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# In your dataset loader:
class CrackDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = apply_clahe(img)
        img = cv2.resize(img, (256, 256))
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.
        return img, mask

    def __len__(self):
        return len(self.image_paths)
