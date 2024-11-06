import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import random


class CrackDataset(Dataset):
    def __init__(self, image_dir, json_dir=None, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform
        self.is_training = json_dir is not None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        sample = {'image': image}
        
        if self.is_training:
            json_name = os.path.join(self.json_dir, self.image_files[idx].replace('.jpg', '.json'))
            try:
                with open(json_name, 'r') as f:
                    annotations = json.load(f)
                    mask = self.create_mask(image.shape, annotations)
                    sample['mask'] = mask
            except FileNotFoundError:
                print(f"Annotation file not found for {img_name}")
                sample['mask'] = np.zeros(image.shape, dtype=np.uint8)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def create_mask(self, image_shape, annotations):
        mask = np.zeros(image_shape, dtype=np.uint8)
        for annotation in annotations['shapes']:
            points = np.array(annotation['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        return mask


def get_dataloaders(train_img_dir, train_json_dir, val_img_dir, val_json_dir=None, batch_size=4):
    train_dataset = CrackDataset(train_img_dir, train_json_dir)
    val_dataset = CrackDataset(val_img_dir, val_json_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def generate_masks(image):
    image_uint8 = image.astype(np.uint8)
    blurred = cv2.GaussianBlur(image_uint8, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned_mask


def apply_dbscan(mask, eps=2, min_samples=5):
    y, x = np.where(mask > 0)  # Get coordinates of non-zero pixels
    points = np.column_stack((x, y))
    
    if len(points) == 0:
        return mask  # No points to cluster, return original mask
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    
    clustered_mask = np.zeros(mask.shape, dtype=np.uint8)
    for label in set(labels):
        if label == -1:
            continue  # Skip noise points
        mask_points = points[labels == label]
        for point in mask_points:
            clustered_mask[point[1], point[0]] = 1
    return clustered_mask


def evaluate_masks(true_masks, pred_masks):
    iou_scores = []
    for true_mask, pred_mask in zip(true_masks, pred_masks):
        true_flat = (true_mask.flatten() > 0).astype(np.uint8)
        pred_flat = (pred_mask.flatten() > 0).astype(np.uint8)
        iou = jaccard_score(true_flat, pred_flat)
        iou_scores.append(iou)
    return np.mean(iou_scores)


def visualize_results(image, true_mask, pred_mask, cleaned_mask, clustered_mask, title_prefix='Original'):
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title(f'{title_prefix} Image')
    ax[1].imshow(true_mask, cmap='gray')
    ax[1].set_title('True Mask')
    ax[2].imshow(pred_mask, cmap='gray')
    ax[2].set_title('Predicted Mask')
    ax[3].imshow(cleaned_mask, cmap='gray')
    ax[3].set_title('Cleaned Mask')
    ax[4].imshow(clustered_mask, cmap='gray')
    ax[4].set_title('DBSCAN Mask')
    plt.show()


# Paths for images and annotations (your file path)
train_img_dir = r''
train_json_dir = r''
val_img_dir = r''
val_json_dir = r''

train_loader, val_loader = get_dataloaders(train_img_dir, train_json_dir, val_img_dir, val_json_dir)

# Training Data Evaluation
train_iou_scores = []

for i, sample in enumerate(train_loader):
    if i >= 20:  # Only evaluating on 20 samples
        break
    image = sample['image'][0].numpy()
    true_mask_train = sample['mask'][0].numpy()
    pred_mask_train = generate_masks(image)
    cleaned_mask_train = clean_mask(pred_mask_train)
    dbscan_mask_train = apply_dbscan(cleaned_mask_train)
    
    # Evaluate IoU between true and DBSCAN-processed mask
    true_flat = (true_mask_train.flatten() > 0).astype(np.uint8)
    dbscan_flat = (dbscan_mask_train.flatten() > 0).astype(np.uint8)
    iou_score = jaccard_score(true_flat, dbscan_flat)
    train_iou_scores.append(iou_score)
    
    visualize_results(image, true_mask_train, pred_mask_train, cleaned_mask_train, dbscan_mask_train, title_prefix='Training')

# Calculate mean IoU for Training Data
mean_iou_train = np.mean(train_iou_scores)
print(f'Mean IoU on Training Data: {mean_iou_train}')

# Validation Data Evaluation
val_iou_scores = []

for i, sample in enumerate(val_loader):
    if i >= 20:  # Only evaluating on 20 samples
        break
    image_val = sample['image'][0].numpy()
    true_mask_val = sample['mask'][0].numpy()
    pred_mask_val = generate_masks(image_val)
    cleaned_mask_val = clean_mask(pred_mask_val)
    dbscan_mask_val = apply_dbscan(cleaned_mask_val)
    
    # Evaluate IoU between true and DBSCAN-processed mask
    true_flat = (true_mask_val.flatten() > 0).astype(np.uint8)
    dbscan_flat = (dbscan_mask_val.flatten() > 0).astype(np.uint8)
    iou_score = jaccard_score(true_flat, dbscan_flat)
    val_iou_scores.append(iou_score)
    
    visualize_results(image_val, true_mask_val, pred_mask_val, cleaned_mask_val, dbscan_mask_val, title_prefix='Validation')

# Calculate mean IoU for Validation Data
mean_iou_val = np.mean(val_iou_scores)
print(f'Mean IoU on Validation Data: {mean_iou_val}')


