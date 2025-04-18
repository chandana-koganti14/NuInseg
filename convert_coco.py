import os
import cv2
import numpy as np
import json
import shutil
from sklearn.utils import shuffle

def get_image_mask_pairs(data_dir):
    image_paths = []
    mask_paths = []
    
    for root, _, files in os.walk(data_dir):
        if 'tissue images' in root:
            for file in files:
                if file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    mask_path = os.path.join(
                        root.replace('tissue images', 'label masks modify'),
                        file.replace('.png', '.tif')
                    )
                    if os.path.exists(mask_path):
                        image_paths.append(image_path)
                        mask_paths.append(mask_path)
    return image_paths, mask_paths

def mask_to_polygons(mask, epsilon=1.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:
            poly = contour.reshape(-1).tolist()
            if len(poly) > 4:
                polygons.append(poly)
    return polygons

def process_data(image_paths, mask_paths, output_dir):
    annotations = []
    images = []
    image_id = 0
    ann_id = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        image_id += 1
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Copy image to output directory
        shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
        
        images.append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "height": img.shape[0],
            "width": img.shape[1]
        })
        
        unique_values = np.unique(mask)
        for value in unique_values:
            if value == 0:
                continue
            
            object_mask = (mask == value).astype(np.uint8) * 255
            polygons = mask_to_polygons(object_mask)
            
            for poly in polygons:
                ann_id += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [poly],
                    "area": cv2.contourArea(np.array(poly).reshape(-1, 2)),
                    "bbox": list(cv2.boundingRect(np.array(poly).reshape(-1, 2))),
                    "iscrowd": 0
                })
    
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "Nuclei"}]
    }
    
    with open(os.path.join(output_dir, 'coco_annotations.json'), 'w') as f:
        json.dump(coco_output, f)

def main():
    data_dir = 'dataset'
    output_dir = 'COCO_output'
    
    # Get all image and mask paths
    image_paths, mask_paths = get_image_mask_pairs(data_dir)
    
    # Shuffle data to randomize
    image_paths, mask_paths = shuffle(image_paths, mask_paths, random_state=42)
    
    total_images = len(image_paths)
    assert total_images >= 665, f"Expected at least 665 images, got {total_images}"
    
    # Define exact split counts
    train_count = 465
    val_count = 133
    test_count = 72
    
    # Split the data explicitly
    train_img = image_paths[:train_count]
    train_mask = mask_paths[:train_count]
    
    val_img = image_paths[train_count:train_count+val_count]
    val_mask = mask_paths[train_count:train_count+val_count]
    
    test_img = image_paths[train_count+val_count:train_count+val_count+test_count]
    test_mask = mask_paths[train_count+val_count:train_count+val_count+test_count]
    
    # Create output folders
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Process each split
    process_data(train_img, train_mask, os.path.join(output_dir, 'train'))
    process_data(val_img, val_mask, os.path.join(output_dir, 'val'))
    process_data(test_img, test_mask, os.path.join(output_dir, 'test'))
    
    print(f"Data split into train ({len(train_img)}), val ({len(val_img)}), test ({len(test_img)})")

if __name__ == '__main__':
    main()
