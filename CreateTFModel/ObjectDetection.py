import json, keras_cv
import os
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict
import optuna
import tensorflow as tf
import pandas as pd
import os
from keras_cv.metrics import MeanAveragePrecision  # type: ignore
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

def get_augmentation_pipeline(bounding_box_format="xyxy"):
    return keras_cv.layers.Augmenter([
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format=bounding_box_format),
        keras_cv.layers.RandomContrast(factor=0.2),
        keras_cv.layers.RandomZoom(height_factor=0.1, width_factor=0.1, bounding_box_format=bounding_box_format),
        keras_cv.layers.RandomRotation(factor=0.05, bounding_box_format=bounding_box_format),
        keras_cv.layers.RandomTranslation(0.1, 0.1, bounding_box_format=bounding_box_format)
    ])

def ObjectDetectionStudy(
    size: str,
    Hypmode: str,
    annotation_path: str,
    image_dir: str,
    trials: int = 100,
    log_csv_path: str = "detection_trials_log.csv"
):

    data, class_map = parse_coco_annotations(annotation_path, image_dir)
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
        # ---------------- Hyperparameters ---------------- #
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        use_aug = trial.suggest_categorical("use_augmentation", [True, False])
        val_split = 0.2

        if size in ["medium", "large"] and Hypmode == "full":
            val_split = trial.suggest_categorical("val_split", strat_val_options)

        epochs = 10 if Hypmode in ["min", "moderate"] else 20

        # ---------------- Data Split ---------------- #
        train_data, val_data = stratified_split(data, val_split)

        # ---------------- TF Dataset ---------------- #
        def format_sample(sample):
            img = tf.io.read_file(sample['image_path'])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (512, 512))
            return {
                "images": img,
                "bounding_boxes": {
                    "boxes": tf.convert_to_tensor(sample['boxes'], dtype=tf.float32),
                    "classes": tf.convert_to_tensor(sample['labels'], dtype=tf.int32)
                }
            }

        train_ds = tf.data.Dataset.from_generator(lambda: (format_sample(x) for x in train_data),
                                                  output_signature={
                                                      "images": tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
                                                      "bounding_boxes": {
                                                          "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                          "classes": tf.TensorSpec(shape=(None,), dtype=tf.int32)
                                                      }
                                                  }).batch(batch_size)

        val_ds = tf.data.Dataset.from_generator(lambda: (format_sample(x) for x in val_data),
                                                output_signature={
                                                    "images": tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
                                                    "bounding_boxes": {
                                                        "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                        "classes": tf.TensorSpec(shape=(None,), dtype=tf.int32)
                                                    }
                                                }).batch(batch_size)

        if use_aug:
            aug_layer = get_augmentation_pipeline()
            train_ds = train_ds.map(lambda x: aug_layer(x))

        # ---------------- Model ---------------- #
        model = get_detection_model(detector_type, backbone, (512, 512, 3), num_classes)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=None,
            metrics=[MeanAveragePrecision(bounding_box_format="xyxy", name="mAP_0.5"),
                     MeanAveragePrecision(bounding_box_format="xyxy", iou_threshold=0.75, name="mAP_0.5_0.95")]
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

    # ---------------- Bi-level Optimization ---------------- #
    for detector_type, backbone in selected_pairs:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, detector_type, backbone), n_trials=trials)

def main():
    # Customize these paths
    annotation_path = "path/to/annotations.json"
    image_dir = "path/to/images"
    
    ObjectDetectionStudy(
        size="medium",  # Options: 'small', 'medium', 'large'
        Hypmode="full",  # Options: 'min', 'moderate', 'full'
        annotation_path=annotation_path,
        image_dir=image_dir,
        trials=50,  # Set number of trials
        log_csv_path="detection_trials_log.csv"
    )

if __name__ == "__main__":
    main()
