import os, zipfile, tarfile,  argparse, yaml, re, warnings
from collections import Counter
from GetModel import load_model
from GetDataset import ReadData, DatasetCategory
warnings.filterwarnings("ignore")
from CreateTFModel.createtfmodel import CreateTFModel
from CreateTorchModel.createtorchmodel import CreateTorchModel
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. TF model formats won't be supported.")

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    try:
        import torchvision.models as models
        TORCHVISION_AVAILABLE = True
    except ImportError:
        TORCHVISION_AVAILABLE = False
        print("TorchVision not available. Some model detection features will be limited.")
    
    try:
        import torchaudio.models as audio_models
        TORCHAUDIO_AVAILABLE = True
    except ImportError:
        TORCHAUDIO_AVAILABLE = False
        print("TorchAudio not available. Audio model detection features will be limited.")

except ImportError:
    PYTORCH_AVAILABLE = False
    TORCHVISION_AVAILABLE = False
    TORCHAUDIO_AVAILABLE = False
    print("PyTorch not available. PyTorch model formats won't be supported.")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available. ONNX model formats won't be supported.")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Initialize argument parser
parser = argparse.ArgumentParser(description="Hyperparameter Tuning Script")

# Add arguments
parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
#parser.add_argument("--hypmode", type=str, default="Min", help="Min/Moderate/Full")
#parser.add_argument("--mode", type=str, default="Min", help="Goal: Minimize Loss or Maximize Accuracy")
#parser.add_argument("--trails", type=int, default=100, help="Total Number of Trials(default:100)")

# Parse arguments
args, unknown = parser.parse_known_args()


if not os.path.exists(args.config):
    raise FileNotFoundError(f"Error: {args.config} not found!")

try:
    with open(args.config, "r") as file:
        data = yaml.safe_load(file)  # Using safe_load() to avoid security risks
except yaml.YAMLError as e:
    print(f"Error reading YAML file: {e}")

modelpath = data["Model"]
datasetpath = data["Dataset"]
Task = data["Task"].lower()
Hypmode = data["Hypmode"].lower()
if Hypmode not in ["min", "moderate", "full"]:
    print("Invalid Hypmode. Goint with default: Min")
trials = data["Trials"]
if not isinstance(trials, int) or trials <= 0:
    print("Invalid number of trials. Setting to default: 100")
    trials = 100

if not os.path.exists(datasetpath):
    raise FileNotFoundError(f"Error: {datasetpath} not found!")
else: 
    data, label, count = ReadData(datasetpath, Task)
    size, imbalance, class_imbalance = DatasetCategory(Task, count)
    if size == "Insufficient":
        print("Error: Insufficient data for training.")
        print("Exiting...")
        exit()
    """elif size == "Too Small":
        print("Warning: Dataset is too small for training.")
        check = input("Do you want to continue? (y/n): ").lower()
        if check == "n":
            print("Exiting...")
            exit()
        elif check == "y":
            print("Continuing with the training...")
        else:
            print("Invalid input. Exiting...")
            exit()"""
    if imbalance:
        extreme = False
        for key,value in class_imbalance.items():
            if value > 60 or value < -60:
                extreme = True
                print(f"Class {key} has {value}% samples of imbalance w.r.t. the average samples per class.")
        if extreme:
            print("Warning: Dataset is imbalanced.")
            print("This may lead to overfitting.")
            check = input("Do you want to continue? (y/n): ").lower()
            if check == "n":
                print("Exiting...")
                exit()
            elif check == "y":
                print("Continuing with the training...")
            else:
                print("Invalid input. Exiting...")
                exit()
        else:
            check = input("Do you want to continue? (y/n): ").lower()
            if check == "n":
                print("Exiting...")
                exit()
            elif check == "y":
                print("Continuing with the training...")
            else:
                print("Invalid input. Exiting...")
                exit()

if not os.path.exists(modelpath):
    print(f"Error: {modelpath} not found!")
    print("Creating a new model...")
    framework = "tensorflow"
    if framework == "tensorflow" or "keras" or "keras/tensorflow" or "tensorflow/keras" or "tf":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install TensorFlow to use this framework.")
        
        # Validate dataset format
        """        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise ValueError("Data must be a list of dictionaries with keys 'image_path', 'boxes', and 'labels'")

        # Validate label format
        if not isinstance(label, dict):
            raise ValueError("Label must be a dictionary mapping category IDs to names")
"""
        # Pass data to CreateTFModel
        CreateTFModel(Task, Hypmode, size, imbalance, class_imbalance, data, label, count, trials)
    elif framework == "pytorch" or "torch":
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch to use this framework.")
        CreateTorchModel(Task,Hypmode,size, imbalance, class_imbalance, data, label, count, trials)
    elif framework == "onnx":
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is not available. Please install ONNX to use this framework.")
        # Add ONNX model creation code here
        # Example: model = create_onnx_model()
    else:
        raise ValueError(f"Unsupported framework: {framework}")

#model, modeltype = load_model(modelpath)
