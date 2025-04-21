import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.models as models
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import warnings
from typing import Tuple, Dict, List, Callable, Any, Optional

# Final 5 Backbones with Input Sizes
BACKBONE_CONFIGS = {
    "Too Small": [
        (models.mobilenet_v2, (224, 224)),
        (models.mobilenet_v3_large, (224, 224)),
        (models.efficientnet_b0, (224, 224)),
        (models.densenet121, (224, 224)),
        (models.resnet50, (224, 224))
    ],
    "Small": [
        (models.mobilenet_v2, (224, 224)),
        (models.mobilenet_v3_large, (224, 224)),
        (models.efficientnet_b0, (224, 224)),
        (models.densenet121, (224, 224)),
        (models.resnet50, (224, 224))
    ],
    "medium": [
        (models.resnet50, (224, 224)),
        (models.efficientnet_b3, (224, 224)),
        (models.densenet121, (224, 224)),
        (models.mobilenet_v2, (224, 224)),
        (models.inception_v3, (299, 299))
    ],
    "large": [
        (models.resnet101, (224, 224)),
        (models.resnet152, (224, 224)),
        (models.efficientnet_b4, (224, 224)),
        (models.efficientnet_b7, (224, 224)),
        (models.inception_v3, (299, 299))
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
def configure_device():
    # Check if GPU is available and set PyTorch device accordingly
    if torch.cuda.is_available():
        print("GPU is available!")
        device = torch.device("cuda")
    else:
        print("GPU not available, using CPU.")
        device = torch.device("cpu")
    return device


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


class ImageClassificationModel(nn.Module):
    def __init__(self, base_model_fn, num_classes, dense_units, dropout_rate, pretrained=True):
        super().__init__()
        # Get the base model without classifier
        self.base_model = base_model_fn(pretrained=pretrained)
        
        # Get the number of features from the base model
        if hasattr(self.base_model, 'fc'):
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove final fully connected layer
        elif hasattr(self.base_model, 'classifier'):
            # For models like DenseNet, MobileNet
            if isinstance(self.base_model.classifier, nn.Sequential):
                num_features = self.base_model.classifier[0].in_features
            else:
                num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError("Unsupported base model architecture")
        
        # Create new classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, num_classes)
        )
    
    def forward(self, x):
        x = self.base_model(x)
        return self.classifier(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), correct / total


def create_objective(model_fn, input_shape, data, labels, Hypmode, size, imbalance=False,
                     class_imbalance=None, class_counts=None, max_batch_size=64):
    """
    Returns an Optuna objective function tailored to dataset size and hyperparameter mode.
    
    Parameters:
    - model_fn: Callable model backbone function from torchvision models.
    - input_shape: Tuple[int, int] — expected input shape of the model.
    - data, labels: Dataset arrays.
    - Hypmode: One of ['minimal', 'moderate', 'full'].
    - size: One of ['too_small', 'medium', 'large'].
    - imbalance: Optional handling for imbalanced classes.
    """
    
    def objective(trial):
        # Get number of unique classes
        if isinstance(labels[0], str):
            unique_labels = np.unique(labels)
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_to_idx[label] for label in labels])
            num_classes = len(unique_labels)
        else:
            numeric_labels = labels
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
            data, numeric_labels, test_size=val_split, stratify=numeric_labels, random_state=42
        )
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # --- Model Build ---
        model = ImageClassificationModel(
            base_model_fn=model_fn,
            num_classes=num_classes,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            pretrained=True
        ).to(device)
        
        # Freeze base model if not in full mode
        if Hypmode != "full":
            for param in model.base_model.parameters():
                param.requires_grad = False
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # --- Training ---
        best_val_acc = 0
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model state
        model.load_state_dict(best_model_state)
        # Final evaluation
        _, final_val_acc = validate(model, val_loader, criterion, device)
        
        return final_val_acc
    
    return objective


def ImageClassificationStudy(size: str, Hypmode, data, labels, imbalance: bool = False, 
                            class_imbalance: dict = None, class_counts: dict = None, 
                            trials: int = 100):
    """
    Main controller for selecting backbones and launching hyperparameter tuning.
    """
    if size not in BACKBONE_CONFIGS:
        raise ValueError(f"Unsupported size '{size}'. Choose from: {list(BACKBONE_CONFIGS.keys())}")
    
    candidate_backbones = BACKBONE_CONFIGS[size]
    
    # Convert Hypmode string to dict format if it's a string
    if isinstance(Hypmode, str):
        Hypmode_dict = {"mode": Hypmode, "n_trials": trials // len(candidate_backbones)}
    else:
        Hypmode_dict = Hypmode
    
    # If conditions met, do bi-level optimization
    if size in ["medium", "large"] and Hypmode_dict.get("mode", Hypmode) == "full":
        best_score = -np.inf
        best_model_name = None
        best_trial = None
        
        for model_fn, input_shape in candidate_backbones:
            print(f"\n🔍 Running inner optimization for backbone: {model_fn.__name__}")
            
            # Resize data to match input shape expected by the model
            resized_data = np.array([
                np.array(torch.nn.functional.interpolate(
                    torch.from_numpy(img[None]).permute(0, 3, 1, 2).float(),
                    size=input_shape
                ).permute(0, 2, 3, 1).numpy()[0]) 
                for img in data
            ])
            
            # Define objective with the current backbone
            objective = create_objective(
                model_fn=model_fn,
                input_shape=input_shape,
                data=resized_data,
                labels=labels,
                Hypmode=Hypmode_dict.get("mode", Hypmode),
                size=size,
                imbalance=imbalance,
                class_imbalance=class_imbalance,
                class_counts=class_counts
            )
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=Hypmode_dict.get("n_trials", 10))
            
            score = study.best_value
            print(f"✅ Backbone {model_fn.__name__} achieved val accuracy: {score:.4f}")
            
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
            
            # Resize data to match input shape expected by the model
            resized_data = np.array([
                np.array(torch.nn.functional.interpolate(
                    torch.from_numpy(img[None]).permute(0, 3, 1, 2).float(),
                    size=input_shape
                ).permute(0, 2, 3, 1).numpy()[0]) 
                for img in data
            ])
            
            objective = create_objective(
                model_fn=model_fn,
                input_shape=input_shape,
                data=resized_data,
                labels=labels,
                Hypmode=Hypmode_dict.get("mode", Hypmode),
                size=size,
                imbalance=imbalance,
                class_imbalance=class_imbalance,
                class_counts=class_counts
            )
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=Hypmode_dict.get("n_trials", 10))
            
            print(f"     Backbone {model_fn.__name__} achieved val accuracy: {study.best_value:.4f}")


def main():
    # Dummy data example
    dummy_data = np.random.rand(100, 300, 300, 3)  # Oversized, will be resized
    dummy_labels = np.random.choice(['cat', 'dog', 'bird'], 100)

    ImageClassificationStudy(
        size="small",
        Hypmode="moderate",
        data=dummy_data,
        labels=dummy_labels
    )


if __name__ == "__main__":
    main()