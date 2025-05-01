import json, keras_cv
import os
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict
import optuna
import tensorflow as tf
import pandas as pd
import os
from keras_cv.api.metrics import BoxCOCOMetrics  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from datetime import datetime

def parse_coco_annotations(annotation_path, image_dir):
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
    # Stratify by the most frequent class in each image
    stratify_labels = [max(set(d['labels']), key=d['labels'].count) for d in data]
    train_data, val_data = train_test_split(data, test_size=val_split, stratify=stratify_labels, random_state=seed)
    return train_data, val_data


def get_detection_model(detector_type, backbone, input_shape, num_classes):
    if detector_type.lower() == "faster_rcnn":
        return keras_cv.models.FasterRCNN(
            backbone=backbone,
            input_shape=input_shape,
            num_classes=num_classes,
            bounding_box_format='xyxy'
        )
    
    elif detector_type.lower() == "ssd":
        return keras_cv.models.SSD(
            backbone=backbone,
            input_shape=input_shape,
            num_classes=num_classes,
            bounding_box_format='xyxy'
        )

    elif detector_type.lower() == "efficientdet":
        return keras_cv.models.EfficientDet(
            backbone=backbone,
            input_shape=input_shape,
            num_classes=num_classes,
            bounding_box_format='xyxy'
        )

    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")


def load_object_detection_data(path):
    """
    Load object detection data from COCO-format JSON file.
    
    Args:
        path: Path to the directory containing 'label.json' and 'images' folder
        
    Returns:
        data: List of dictionaries with image paths and annotations
        category_id_to_name: Dictionary mapping category IDs to names
        class_counts: Dictionary of class counts for dataset statistics
    """
    from collections import defaultdict
    import os
    import json
    
    json_path = os.path.join(path, 'label.json')
    images_dir = os.path.join(path, 'images')

    with open(json_path, 'r') as f:
        label = json.load(f)

    # Build lookup tables
    id_to_filename = {img['id']: img['file_name'] for img in label['images']}
    id_to_annotations = defaultdict(list)
    class_counts = defaultdict(int)
    category_id_to_name = {cat['id']: cat['name'] for cat in label['categories']}

    # Populate the annotations dictionary - this was missing in your code
    for ann in label['annotations']:
        id_to_annotations[ann['image_id']].append(ann)
        class_counts[category_id_to_name[ann['category_id']]] += 1

    data = []
    
    for image_id, file_name in id_to_filename.items():
        image_path = os.path.join(images_dir, file_name)
        if os.path.exists(image_path):
            boxes = []
            labels = []
            
            for ann in id_to_annotations[image_id]:
                x, y, w, h = ann['bbox']
                # Convert from COCO format (xywh) to xyxy format used by keras-cv
                boxes.append([x, y, x + w, y + h])  
                labels.append(ann['category_id'])
                
            if boxes:  # Only add images that have annotations
                data.append({
                    'image_path': image_path,
                    'boxes': boxes,
                    'labels': labels
                })
    
    return data, category_id_to_name, class_counts


def create_dataset(data, batch_size, is_training=False):
    """Create tf.data.Dataset from detection data"""
    
    def format_sample(sample):
        img = tf.io.read_file(sample['image_path'])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (512, 512))
        
        # Ensure boxes are within image bounds after resize
        boxes = tf.convert_to_tensor(sample['boxes'], dtype=tf.float32)
        image_size = tf.constant([512.0, 512.0, 512.0, 512.0])
        boxes = tf.minimum(boxes, image_size)
        boxes = tf.maximum(boxes, 0.0)
        
        return {
            "images": img,
            "bounding_boxes": {
                "boxes": boxes,
                "classes": tf.convert_to_tensor(sample['labels'], dtype=tf.int32)
            }
        }
    
    def gen():
        for sample in data:
            yield format_sample(sample)
    
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "images": tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
            "bounding_boxes": {
                "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                "classes": tf.TensorSpec(shape=(None,), dtype=tf.int32)
            }
        }
    )
    
    # Batch, prefetch for performance
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def ObjectDetectionStudy(
    size: str,
    Hypmode: str,
    data: list,
    class_map: dict,
    imbalance: bool = False,
    class_imbalance: dict = None,
    class_counts: dict = None,
    trials: int = 100,
    log_csv_path: str = "detection_trials_log.csv"
):
    """
    Run an object detection hyperparameter study using optuna.
    
    Args:
        size: Model size ('small', 'medium', or 'large')
        Hypmode: Hyperparameter optimization mode ('min', 'moderate', or 'full')
        data: List of dictionaries with image paths and annotations
        class_map: Dictionary mapping category IDs to names
        trials: Number of optimization trials
        log_csv_path: Path to save trial results
    """
    
    # No need to call parse_coco_annotations anymore
    num_classes = len(class_map)
    dataset_size = len(data)
    strat_val_options = [0.15, 0.25, 0.35]

    detector_backbones = {
        "small": [("ssd", "mobilenetv2"), ("ssd", "vgg16")],
        "medium": [("faster_rcnn", "resnet50"), ("efficientdet", "efficientnetb0")],
        "large": [("faster_rcnn", "resnet101"), ("efficientdet", "efficientnetb3")]
    }

    # Determine detector-backbone pairs
    if size in ["too small", "small"]:
        selected_pairs = detector_backbones["small"]
    elif size == "medium":
        selected_pairs = detector_backbones["medium"]
    else:
        selected_pairs = detector_backbones["large"]

    def objective(trial, detector_type, backbone):
        try:
            # Validate data format inside the objective function
            if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
                raise ValueError("Data must be a list of dictionaries with keys 'image_path', 'boxes', and 'labels'")
            
            # ---------------- Hyperparameters ---------------- #
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
            use_aug = trial.suggest_categorical("use_augmentation", [True, False])
            val_split = 0.2

            if size in ["medium", "large"] and Hypmode == "full":
                val_split = trial.suggest_categorical("val_split", strat_val_options)

            epochs = 10 if Hypmode in ["min", "moderate"] else 20

            # ---------------- Data Split ---------------- #
            from sklearn.model_selection import train_test_split
            
            # Stratify by the most frequent class in each image
            stratify_labels = [max(set(d['labels']), key=d['labels'].count) for d in data]
            train_data, val_data = train_test_split(data, test_size=val_split, 
                                                    stratify=stratify_labels, random_state=69)

            # ---------------- TF Dataset ---------------- #
            # Create datasets without augmentation
            train_ds = create_dataset(train_data, batch_size, is_training=True)
            val_ds = create_dataset(val_data, batch_size)

            # ---------------- Model ---------------- #
            model = get_detection_model(detector_type, backbone, (512, 512, 3), num_classes)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=None,
                metrics=[BoxCOCOMetrics(bounding_box_format="xyxy")]
            )

            # ---------------- Callbacks ---------------- #
            callbacks = []
            if Hypmode == "full":
                callbacks.append(EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True))

            # ---------------- Training ---------------- #
            history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=0)

            metrics = model.evaluate(val_ds, return_dict=True, verbose=0)

            # ---------------- Logging ---------------- #
            log_dict = {
                "trial_number": trial.number,
                "detector_type": detector_type,
                "backbone": backbone,
                "learning_rate": lr,
                "batch_size": batch_size,
                "val_split": val_split,
                "epochs": epochs,
                "use_augmentation": use_aug,
                "mAP_0.5": metrics["mAP_0.5"],
                "mAP_0.5_0.95": metrics["mAP_0.5_0.95"],
            }

            if os.path.exists(log_csv_path):
                df_log = pd.read_csv(log_csv_path)
                df_log = pd.concat([df_log, pd.DataFrame([log_dict])], ignore_index=True)
            else:
                df_log = pd.DataFrame([log_dict])
            df_log.to_csv(log_csv_path, index=False)

            return metrics["mAP_0.5_0.95"]  # Can change to mAP@0.5 if preferred
        
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            # Log error to help with debugging
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("trial_error_log.txt", "a") as f:
                f.write(f"{timestamp} - Trial {trial.number}: {str(e)}\n")
            raise e

    # ---------------- Bi-level Optimization ---------------- #
    for detector_type, backbone in selected_pairs:
        study_name = f"{detector_type}_{backbone}"
        print(f"\nStarting optimization for {study_name}")
        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(lambda trial: objective(trial, detector_type, backbone), n_trials=trials)
        
        # Print best parameters
        print(f"Best trial for {study_name}:")
        print(f"  Value: {study.best_value}")
        print(f"  Params: {study.best_params}")
            
def main():
    """
    Main function to run object detection study with proper data loading
    """
    import os
    
    # Customize these paths
    data_path = "path/to/dataset"
    log_csv_path = "detection_trials_log.csv"
    
    # Load data using the corrected function
    data, category_id_to_name, class_counts = load_object_detection_data(data_path)
    
    # Ensure data is in the correct format
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise ValueError("Data must be a list of dictionaries with keys 'image_path', 'boxes', and 'labels'")
    
    # Print dataset statistics
    print(f"Loaded {len(data)} images with annotations")
    print(f"Class distribution: {class_counts}")
    print(f"Number of classes: {len(category_id_to_name)}")
    
    # Run the object detection study
    ObjectDetectionStudy(
        size="medium",  # Options: 'small', 'medium', 'large'
        Hypmode="full",  # Options: 'min', 'moderate', 'full'
        data=data,  # Pass the data directly instead of paths
        class_map=category_id_to_name,  # Pass the class map
        trials=50,  # Set number of trials
        log_csv_path=log_csv_path
    )

if __name__ == "__main__":
    main()