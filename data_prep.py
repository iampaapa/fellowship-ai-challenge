import os
import shutil
from scipy.io import loadmat
import numpy as np

# Load labels and splits
labels = loadmat('imagelabels.mat')['labels'][0]
splits = loadmat('setid.mat')
train_ids = splits['trnid'][0]
val_ids = splits['valid'][0]
test_ids = splits['tstid'][0]

# Create directories
for split in ['train', 'val', 'test']:
    for class_id in range(1, 103):
        os.makedirs(f'flower_data/{split}/{class_id}', exist_ok=True)

# Move images to appropriate directories
for img_id, label in enumerate(labels, start=1):
    img_name = f'image_{img_id:05d}.jpg'
    src_path = f'jpg/{img_name}'
    
    if img_id in train_ids:
        dst_dir = f'flower_data/train/{label}'
    elif img_id in val_ids:
        dst_dir = f'flower_data/val/{label}'
    elif img_id in test_ids:
        dst_dir = f'flower_data/test/{label}'
    
    shutil.copy(src_path, f'{dst_dir}/{img_name}')

print("Completed Data Prep")