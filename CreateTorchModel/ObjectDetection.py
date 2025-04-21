import json
import os
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, 
    ssd300_vgg16,
    retinanet_resnet50_fpn
)
from torchvision.ops import box_iou
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
import time
from datetime import datetime


def parse_coco_annotations(annotation_path, image_dir):
    """Parse COCO format annotations into a more manageable format."""
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Map image ID to file path
    image_id_to_file = {img['id']: os.path.join(image_dir, img['file_name']) for img in coco_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    data = []
    for image_id, anns in annotations_by_image.items():
        if image_id not in image_id_to_file:
            continue
        image_path = image_id_to_file[image_id]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])
        data.append({
            'image_path': image_path,
            'boxes': boxes,
            'labels': labels
        })
    return data, category_id_to_name


def stratified_split(data, val_split=0.2, seed=69):
    """Split the dataset with stratification based on the most frequent class in each image."""
    stratify_labels = [max(set(d['labels']), key=d['labels'].count) for d in data]
    train_data, val_data = train_test_split(data, test_size=val_split, stratify=stratify_labels, random_state=seed)
    return train_data, val_data


class COCODetectionDataset(Dataset):
    """PyTorch Dataset for object detection data in COCO format."""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")
        
        boxes = torch.as_tensor(sample["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(sample["labels"], dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


class DetectionTransforms:
    """Class to handle transformations for object detection."""
    def __init__(self, size=(512, 512), augment=False):
        self.size = size
        self.augment = augment
        
        # Base transforms for both train and validation
        self.base_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Additional augmentations for training
        if augment:
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(5),
                self.base_transform
            ])
        else:
            self.train_transform = self.base_transform
            
        self.val_transform = self.base_transform
    
    def train_transforms(self):
        return self.train_transform
    
    def val_transforms(self):
        return self.val_transform


def collate_fn(batch):
    """Custom collate function for the DataLoader to handle variable size targets."""
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets


def get_detection_model(detector_type, backbone, num_classes, pretrained=True):
    """Get the appropriate detection model based on type and backbone."""
    if detector_type.lower() == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        # Modify the box predictor to match our number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = nn.modules.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
    elif detector_type.lower() == "ssd":
        model = ssd300_vgg16(pretrained=pretrained)
        # Replace the classifier head
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = [4, 6, 6, 6, 4, 4]
        model.head.classification_head = nn.modules.detection.ssd.SSDClassificationHead(
            in_channels, num_anchors, num_classes
        )
        
    elif detector_type.lower() == "retinanet":  # Using RetinaNet instead of EfficientDet
        model = retinanet_resnet50_fpn(pretrained=pretrained)
        # Modify the classification head for our classes
        model.head.classification_head.num_classes = num_classes
        
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
        
    return model


def mean_average_precision(preds, targets, iou_threshold=0.5):
    """Calculate Mean Average Precision at given IoU threshold."""
    aps = []
    
    for pred, target in zip(preds, targets):
        if len(pred["boxes"]) == 0 or len(target["boxes"]) == 0:
            continue
            
        # Calculate IoU between predicted and ground truth boxes
        iou = box_iou(pred["boxes"], target["boxes"])
        
        # For each predicted box, find the best matching ground truth box
        max_iou, max_idx = torch.max(iou, dim=1)
        
        # Count true positives (IoU > threshold and correct class)
        true_positives = torch.zeros(len(pred["boxes"]))
        false_positives = torch.zeros(len(pred["boxes"]))
        
        for i, (score, label, iou_val, gt_idx) in enumerate(zip(
            pred["scores"], pred["labels"], max_iou, max_idx
        )):
            if iou_val >= iou_threshold and label == target["labels"][gt_idx]:
                true_positives[i] = 1
            else:
                false_positives[i] = 1
                
        # Sort by confidence scores
        indices = torch.argsort(pred["scores"], descending=True)
        true_positives = true_positives[indices]
        false_positives = false_positives[indices]
        
        # Compute cumulative sums
        cumulative_tp = torch.cumsum(true_positives, dim=0)
        cumulative_fp = torch.cumsum(false_positives, dim=0)
        
        # Calculate precision and recall
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = cumulative_tp / len(target["boxes"])
        
        # Calculate AP using precision-recall curve
        # Use all points interpolation
        ap = 0.0
        for r in torch.linspace(0, 1, steps=11):
            if torch.sum(recall >= r) == 0:
                p = 0
            else:
                p = torch.max(precision[recall >= r])
            ap += p / 11
        
        aps.append(ap.item())
    
    return sum(aps) / len(aps) if aps else 0


def evaluate_model(model, data_loader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            
            all_preds.extend(outputs)
            all_targets.extend(targets)
    
    # Calculate metrics
    mAP_50 = mean_average_precision(all_preds, all_targets, iou_threshold=0.5)
    mAP_75 = mean_average_precision(all_preds, all_targets, iou_threshold=0.75)
    
    # Calculate average mAP across thresholds 0.5 to 0.95 (COCO style)
    mAP_range = 0.0
    thresholds = [t/100 for t in range(50, 100, 5)]  # [0.5, 0.55, ..., 0.95]
    for threshold in thresholds:
        mAP_range += mean_average_precision(all_preds, all_targets, iou_threshold=threshold)
    mAP_range /= len(thresholds)
    
    return {
        "mAP_0.5": mAP_50,
        "mAP_0.75": mAP_75,
        "mAP_0.5_0.95": mAP_range
    }


def train_epoch(model, data_loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)


def ObjectDetectionStudy(
    size: str,
    hypmode: str,
    annotation_path: str,
    image_dir: str,
    trials: int = 100,
    log_csv_path: str = "detection_trials_log.csv"
):
    """Main function to run the object detection optimization study."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse data
    data, class_map = parse_coco_annotations(annotation_path, image_dir)
    num_classes = len(class_map) + 1  # +1 for background
    dataset_size = len(data)
    print(f"Dataset size: {dataset_size}, Number of classes: {num_classes}")
    
    strat_val_options = [0.15, 0.25, 0.35]
    
    detector_backbones = {
        "small": [("ssd", "vgg16")],
        "medium": [("faster_rcnn", "resnet50")],
        "large": [("faster_rcnn", "resnet101"), ("retinanet", "resnet50")]
    }
    
    # Determine detector-backbone pairs
    if size in ["too small", "small"]:
        selected_pairs = detector_backbones["small"]
    elif size == "medium":
        selected_pairs = detector_backbones["medium"]
    else:
        selected_pairs = detector_backbones["large"]
        
    def objective(trial, detector_type, backbone):
        # Hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])  # Reduced sizes due to GPU memory
        use_aug = trial.suggest_categorical("use_augmentation", [True, False])
        val_split = 0.2
        
        if size in ["medium", "large"] and hypmode == "full":
            val_split = trial.suggest_categorical("val_split", strat_val_options)
            
        epochs = 10 if hypmode in ["min", "moderate"] else 20
        
        # Data split
        train_data, val_data = stratified_split(data, val_split)
        
        # Transforms
        transforms_handler = DetectionTransforms(size=(512, 512), augment=use_aug)
        
        # Create datasets
        train_dataset = COCODetectionDataset(train_data, transform=transforms_handler.train_transforms())
        val_dataset = COCODetectionDataset(val_data, transform=transforms_handler.val_transforms())
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2
        )
        
        # Create model
        model = get_detection_model(detector_type, backbone, num_classes)
        model.to(device)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Learning rate scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=3,
            verbose=True
        )
        
        # Training loop
        best_mAP = 0.0
        early_stop_counter = 0
        early_stop_patience = 4
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device)
            print(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            metrics = evaluate_model(model, val_loader, device)
            print(f"Validation mAP@0.5: {metrics['mAP_0.5']:.4f}, mAP@0.5-0.95: {metrics['mAP_0.5_0.95']:.4f}")
            
            # Update learning rate
            lr_scheduler.step(train_loss)
            
            # Early stopping check
            if hypmode == "full":
                if metrics['mAP_0.5_0.95'] > best_mAP:
                    best_mAP = metrics['mAP_0.5_0.95']
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    
                if early_stop_counter >= early_stop_patience:
                    print("Early stopping triggered")
                    break
        
        # Final evaluation
        final_metrics = evaluate_model(model, val_loader, device)
        
        # Logging
        log_dict = {
            "trial_number": trial.number,
            "detector_type": detector_type,
            "backbone": backbone,
            "learning_rate": lr,
            "batch_size": batch_size,
            "val_split": val_split,
            "epochs": epoch + 1,  # Actual epochs run
            "use_augmentation": use_aug,
            "mAP_0.5": final_metrics["mAP_0.5"],
            "mAP_0.75": final_metrics["mAP_0.75"],
            "mAP_0.5_0.95": final_metrics["mAP_0.5_0.95"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if os.path.exists(log_csv_path):
            df_log = pd.read_csv(log_csv_path)
            df_log = pd.concat([df_log, pd.DataFrame([log_dict])], ignore_index=True)
        else:
            df_log = pd.DataFrame([log_dict])
        df_log.to_csv(log_csv_path, index=False)
        
        return final_metrics["mAP_0.5_0.95"]
    
    # Bi-level Optimization
    for detector_type, backbone in selected_pairs:
        print(f"\nOptimizing {detector_type} with {backbone} backbone")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, detector_type, backbone), n_trials=trials)
        
        print(f"Best trial for {detector_type}-{backbone}:")
        print(f"  Value: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")


def main():
    # Customize these paths
    annotation_path = "path/to/annotations.json"
    image_dir = "path/to/images"
    
    ObjectDetectionStudy(
        size="medium",  # Options: 'small', 'medium', 'large'
        hypmode="full",  # Options: 'min', 'moderate', 'full'
        annotation_path=annotation_path,
        image_dir=image_dir,
        trials=50,  # Set number of trials
        log_csv_path="detection_trials_log_pytorch.csv"
    )


if __name__ == "__main__":
    main()