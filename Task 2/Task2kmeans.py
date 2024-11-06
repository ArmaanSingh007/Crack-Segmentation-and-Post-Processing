import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt



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
    reshaped_image = image.reshape((-1, 1))
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(reshaped_image)
    clustered = kmeans.labels_.reshape(image.shape)
    
    # Convert to numpy array for mean calculation
    image = np.array(image)
    
    # Assume the cluster with the smaller mean intensity is the crack
    crack_cluster = 0 if np.mean(image[clustered == 0]) < np.mean(image[clustered == 1]) else 1
    
    mask = (clustered == crack_cluster).astype(np.uint8)
    return mask


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask


def evaluate_masks(true_masks, pred_masks):
    iou_scores = [jaccard_score(true_mask.flatten(), pred_mask.flatten(), average='binary') 
                  for true_mask, pred_mask in zip(true_masks, pred_masks)]
    return np.mean(iou_scores)


def visualize_results(image, true_mask, pred_mask, cleaned_mask, title_prefix='Original'):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title(f'{title_prefix} Image')
    ax[1].imshow(true_mask, cmap='gray')
    ax[1].set_title('True Mask')
    ax[2].imshow(pred_mask, cmap='gray')
    ax[2].set_title('Predicted Mask')
    ax[3].imshow(cleaned_mask, cmap='gray')
    ax[3].set_title('Cleaned Mask')
    plt.show()


# Example usage (your file path)
train_img_dir = r''
train_json_dir = r''
val_img_dir = r''
val_json_dir = r''

train_loader, val_loader = get_dataloaders(train_img_dir, train_json_dir, val_img_dir, val_json_dir)

true_masks_train = []
pred_masks_train = []
cleaned_masks_train = []

true_masks_val = []
pred_masks_val = []
cleaned_masks_val = []

train_iou_scores = []
val_iou_scores = []

# Training Data Evaluation
for i, sample in enumerate(train_loader):
    if i >= 20:  # only evaluating on 20 samples
        break
    image = sample['image'][0].numpy()  # Extract single image from batch and convert to numpy
    true_mask_train = sample['mask'][0].numpy()  # Extract single mask from batch and convert to numpy
    pred_mask_train = generate_masks(image)
    cleaned_mask_train = clean_mask(pred_mask_train)
    
    true_masks_train.append(true_mask_train)
    pred_masks_train.append(pred_mask_train)
    cleaned_masks_train.append(cleaned_mask_train)
    
    iou_score = jaccard_score(true_mask_train.flatten(), pred_mask_train.flatten(), average='binary')
    train_iou_scores.append(iou_score)
    
    visualize_results(image, true_mask_train, pred_mask_train, cleaned_mask_train, title_prefix='Training')

# Calculate mean IoU for Training Data
mean_iou_train = np.mean(train_iou_scores)
print(f'Mean IoU on Training Data: {mean_iou_train}')

# Validation Data Evaluation
for i, sample in enumerate(val_loader):
    if i >= 20:  # only evaluating on 20 samples
        break
    image_val = sample['image'][0].numpy()  # Extract single image from batch and convert to numpy
    true_mask_val = sample['mask'][0].numpy()  # Extract single mask from batch and convert to numpy
    pred_mask_val = generate_masks(image_val)
    cleaned_mask_val = clean_mask(pred_mask_val)
    
    true_masks_val.append(true_mask_val)
    pred_masks_val.append(pred_mask_val)
    cleaned_masks_val.append(cleaned_mask_val)
    
    iou_score = jaccard_score(true_mask_val.flatten(), pred_mask_val.flatten(), average='binary')
    val_iou_scores.append(iou_score)
    
    visualize_results(image_val, true_mask_val, pred_mask_val, cleaned_mask_val, title_prefix='Validation')

# Calculate mean IoU for Validation Data
mean_iou_val = np.mean(val_iou_scores)
print(f'Mean IoU on Validation Data: {mean_iou_val}')

# Plot IoU scores for Training and Validation
plt.figure(figsize=(10, 5))
plt.plot(train_iou_scores, label='Training IoU Scores')
plt.plot(val_iou_scores, label='Validation IoU Scores')
plt.xlabel('Sample Index')
plt.ylabel('IoU Score')
plt.title('IoU Scores for Training and Validation Data')
plt.legend()
plt.show()

# Compare Validation Data with Predicted and Cleaned Masks from Training Data
for i in range(min(20, len(true_masks_val))):
    visualize_results(image_val, true_masks_val[i], pred_masks_train[i], cleaned_masks_train[i], title_prefix='Validation')



