import tensorflow as tf
import optuna
import numpy as np
from tensorflow.keras.applications import ( # type: ignore
    ResNet50, ResNet101,
    EfficientNetB0, EfficientNetB3, EfficientNetB5,
    MobileNetV2, MobileNetV3Large,
    DenseNet121,
    Xception, InceptionV3,
    VGG19, ResNet152,
    EfficientNetB4, EfficientNetB7
)
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
from tensorflow.keras.metrics import SparseCategoricalAccuracy # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os, cv2, glob, warnings, random


# Final 5 Backbones with Input Sizes
BACKBONE_CONFIGS = {
    "Too Small": [
        (MobileNetV2, (224, 224)),
        (MobileNetV3Large, (224, 224)),
        (EfficientNetB0, (224, 224)),
        (DenseNet121, (224, 224)),
        (ResNet50, (224, 224))
    ],
    "Small": [
        (MobileNetV2, (224, 224)),
        (MobileNetV3Large, (224, 224)),
        (EfficientNetB0, (224, 224)),
        (DenseNet121, (224, 224)),
        (ResNet50, (224, 224))
    ],
    "medium": [
        (ResNet50, (224, 224)),
        (EfficientNetB3, (224, 224)),
        (DenseNet121, (224, 224)),
        (Xception, (299, 299)),
        (MobileNetV2, (224, 224))
    ],
    "large": [
        (ResNet101, (224, 224)),
        (ResNet152, (224, 224)),
        (EfficientNetB4, (224, 224)),
        (EfficientNetB7, (224, 224)),
        (InceptionV3, (299, 299))
    ]
}

def create_csv_log_file(log_file):
    columns = [
        "trial_number",
        "learning_rate",
        "batch_size",
        "dropout_rate",
        "dense_units",
        "val_split",
        "epochs",
        "val_accuracy"
    ]
    if not os.path.isfile(log_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(log_file, index=False)

def log_trial_results(
    log_file, trial_number, learning_rate, batch_size,
    dropout_rate, dense_units, val_split, epochs, val_accuracy
):
    row = {
        "trial_number": trial_number,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "dense_units": dense_units,
        "val_split": val_split,
        "epochs": epochs,
        "val_accuracy": val_accuracy
    }
    pd.DataFrame([row]).to_csv(log_file, mode="a", header=False, index=False)

# Device independence: check and use available device
def configure_device(memory_limit_fraction=0.5):
    """
    Configure TensorFlow to use GPU if available and limit memory usage only when GPU is not available.
    
    Args:
        memory_limit_fraction: Fraction of memory to allocate (default is 0.5 for 50%).
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU is available!")
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Error configuring GPU memory: {e}")
    else:
        print("GPU not available, limiting CPU memory usage.")
        # Limit CPU memory usage
        gpus = tf.config.list_physical_devices('CPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(memory_limit_fraction * 1024))]
                )
            except Exception as e:
                print(f"Error configuring CPU memory: {e}")

# Get the available device (GPU or CPU)
device = configure_device()

def get_dense_units_options(num_classes, size):
    if num_classes <= 10:
        options = [64, 128]
    elif num_classes <= 50:
        options = [128, 256]
    else:
        options = [256, 512]

    # Cap units if dataset is small
    if size in ["too_small", "small"]:
        options = [x for x in options if x <= 128]

    return options


def get_dropout_range(size):
    if size in ["too_small", "small"]:
        return (0.4, 0.6)
    elif size == "medium":
        return (0.3, 0.5)
    else:
        return (0.2, 0.4)


def get_learning_rate_range(size):
    if size in ["too_small", "small"]:
        return (1e-5, 5e-4)
    elif size == "medium":
        return (5e-5, 5e-4)
    else:
        return (1e-4, 1e-3)


def get_batch_size_options(max_batch_size):
    return [bs for bs in [16, 32, 64, 128] if bs <= max_batch_size]


def create_objective(model_fn, input_shape, data, labels, Hypmode, size, imbalance=False,
                     class_imbalance=None, class_counts=None, max_batch_size=64):
    """
    Returns an Optuna objective function tailored to dataset size and hyperparameter mode.
    
    Parameters:
    - model_fn: Callable model backbone function from Keras applications.
    - input_shape: Tuple[int, int] — expected input shape of the model.
    - data, labels: Dataset arrays.
    - Hypmode: One of ['minimal', 'moderate', 'full'].
    - size: One of ['too_small', 'medium', 'large'].
    - imbalance: Optional handling for imbalanced classes.
    """
    def objective(trial):
        num_classes = len(np.unique(labels))

        # --- Learning Rate ---
        lr_low, lr_high = get_learning_rate_range(size)
        lr = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)

        # --- Epochs ---
        epochs = 20 if Hypmode == "full" else 10

        # --- Validation Split ---
        if size in ["medium", "large"] and Hypmode == "full":
            val_split = trial.suggest_categorical("val_split", [0.15, 0.2, 0.25, 0.3, 0.35])
        else:
            val_split = 0.2

        # --- Batch Size ---
        if Hypmode in ["moderate", "full"]:
            batch_size_options = get_batch_size_options(max_batch_size)
            batch_size = trial.suggest_categorical("batch_size", batch_size_options)
        else:
            batch_size = 32

        # --- Dropout ---
        if Hypmode == "full":
            dropout_low, dropout_high = get_dropout_range(size)
            dropout_rate = trial.suggest_float("dropout_rate", dropout_low, dropout_high)
        else:
            dropout_rate = 0.5

        # --- Dense Units ---
        if Hypmode in ["moderate", "full"]:
            dense_unit_options = get_dense_units_options(num_classes, size)
            dense_units = trial.suggest_categorical("dense_units", dense_unit_options)
        else:
            dense_units = 128

        # --- Data Split ---
        X_train, X_val, y_train, y_val = train_test_split(
            data, labels, test_size=val_split, stratify=labels, random_state=42
        )

        # --- Model Build ---
        base_model = model_fn(
            include_top=False,
            weights='imagenet',
            input_shape=(input_shape[0], input_shape[1], 3)
        )
        base_model.trainable = Hypmode == "full"

        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            Dense(dense_units, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )

        # --- Training ---
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=0
        )

        val_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
        return val_accuracy
    return objective


def train(data, labels, Hypmode, n_trials=30):
    objective = create_objective(data, labels, Hypmode)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best trial:", study.best_trial.params)
    return study


def load_and_preprocess_images(file_paths, input_shape, batch_size=32):
    """
    Load images from file paths and preprocess them to the required input shape using TensorFlow's tf.data API.
    
    Args:
        file_paths: List of image file paths
        input_shape: Target shape (height, width) for resizing
        batch_size: Number of images to process in a batch
    
    Returns:
        A tf.data.Dataset object with preprocessed images
    """
    def preprocess_image(file_path):
        # Read and decode the image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        
        # Resize to target input shape
        img = tf.image.resize(img, input_shape)
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        return img

    # Create a TensorFlow dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    # Map the preprocessing function with parallel calls
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch to improve performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def dataset_to_numpy(dataset):
    """
    Convert a tf.data.Dataset to NumPy arrays.
    
    Args:
        dataset: A tf.data.Dataset object.
    
    Returns:
        A tuple of NumPy arrays (data, labels).
    """
    data, labels = [], []
    for batch in dataset:
        data.append(batch[0].numpy())
        labels.append(batch[1].numpy())
    return np.concatenate(data), np.concatenate(labels)

def ImageClassificationStudy(size: str, Hypmode: str, data, labels, imbalance: bool = False, 
                           class_imbalance: dict = None, class_counts: dict = None, trials: int = 100, 
                           log_file: str = "image_classification_study_log.csv"):
    """
    Main controller for selecting backbones and launching hyperparameter tuning.
    """
    if size not in BACKBONE_CONFIGS:
        raise ValueError(f"Unsupported size '{size}'. Choose from: {list(BACKBONE_CONFIGS.keys())}")
    
    candidate_backbones = BACKBONE_CONFIGS[size]
    
    # Create the CSV log file if it doesn't exist
    create_csv_log_file(log_file)
    
    # Convert labels to numeric if they're strings
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    
    # Check if data contains file paths
    is_file_paths = isinstance(data, list) and len(data) > 0 and isinstance(data[0], str)

    # If conditions met, do bi-level optimization
    if size in ["medium", "large"] and Hypmode == "full":
        best_score = -np.inf
        best_model_name = None
        best_trial = None

        for model_fn, input_shape in candidate_backbones:
            print(f"\n🔍 Running inner optimization for backbone: {model_fn.__name__}")
            
            # Load and preprocess images if data is file paths
            if is_file_paths:
                print(f"Loading and preprocessing images to shape {input_shape}...")
                processed_data = load_and_preprocess_images(data, input_shape)
            else:
                # If data is already loaded, just resize
                processed_data = tf.image.resize(data, input_shape)

            # Define objective with the current backbone
            objective = create_objective(
                model_fn=model_fn,
                input_shape=input_shape,
                data=processed_data,
                labels=numeric_labels,
                Hypmode=Hypmode,
                size=size,
                imbalance=imbalance,
                class_imbalance=class_imbalance,
                class_counts=class_counts
            )

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=trials)  # Use the `trials` argument directly

            score = study.best_value
            print(f"✅ Backbone {model_fn.__name__} achieved val accuracy: {score:.4f}")

            # Log the best trial for this backbone
            log_trial_results(
                log_file=log_file,
                trial_number=study.best_trial.number,
                learning_rate=study.best_trial.params.get("learning_rate", None),
                batch_size=study.best_trial.params.get("batch_size", None),
                dropout_rate=study.best_trial.params.get("dropout_rate", None),
                dense_units=study.best_trial.params.get("dense_units", None),
                val_split=study.best_trial.params.get("val_split", None),
                epochs=20 if Hypmode == "full" else 10,
                val_accuracy=score
            )

            if score > best_score:
                best_score = score
                best_model_name = model_fn.__name__
                best_trial = study.best_trial

        print(f"\n🏆 Best Backbone: {best_model_name} with accuracy: {best_score:.4f}")
        print(f"Best hyperparameters: {best_trial.params}")

    else:
        # Regular loop over backbones (no bi-level optimization)
        for model_fn, input_shape in candidate_backbones:
            print(f"\n🎯 Running standard optimization for backbone: {model_fn.__name__}")
            
            # Load and preprocess images if data is file paths
            if is_file_paths:
                print(f"Loading and preprocessing images to shape {input_shape}...")
                processed_data = load_and_preprocess_images(data, input_shape)
                processed_data = processed_data.map(lambda x: (x, numeric_labels))  # Add labels to dataset
            else:
                # If data is already loaded, just resize
                processed_data = tf.image.resize(data, input_shape)
            
            # Convert tf.data.Dataset to NumPy arrays if necessary
            if is_file_paths:
                processed_data, numeric_labels = dataset_to_numpy(processed_data)

            # --- Data Split ---
            X_train, X_val, y_train, y_val = train_test_split(
                processed_data, numeric_labels, test_size=0.2, stratify=numeric_labels, random_state=42
            )
            
            # Inspect the shape of one batch from the dataset
            if is_file_paths:
                for batch in processed_data.take(1):
                    print(f"Processed batch shape: {batch.shape}")
            else:
                print(f"Processed data shape: {processed_data.shape}")

            objective = create_objective(
                model_fn=model_fn,
                input_shape=input_shape,
                data=X_train,
                labels=y_train,
                Hypmode=Hypmode,
                size=size,
                imbalance=imbalance,
                class_imbalance=class_imbalance,
                class_counts=class_counts
            )

            # Reset the study for each backbone
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=trials)  # Use the `trials` argument directly

            # Log the best trial for this backbone
            log_trial_results(
                log_file=log_file,
                trial_number=study.best_trial.number,
                learning_rate=study.best_trial.params.get("learning_rate", None),
                batch_size=study.best_trial.params.get("batch_size", None),
                dropout_rate=study.best_trial.params.get("dropout_rate", None),
                dense_units=study.best_trial.params.get("dense_units", None),
                val_split=study.best_trial.params.get("val_split", None),
                epochs=20 if Hypmode == "full" else 10,
                val_accuracy=study.best_value
            )

            print(f"     Backbone {model_fn.__name__} achieved val accuracy: {study.best_value:.4f}")

def main():
    # Dummy data example
    dummy_data = np.random.rand(100, 300, 300, 3)  # Oversized, will be resized
    dummy_labels = np.random.choice(['cat', 'dog', 'bird'], 100)

    study = ImageClassificationStudy(
        size="small",
        Hypmode="moderate",
        data=dummy_data,
        labels=dummy_labels
    )

if __name__ == "__main__":
    main()
