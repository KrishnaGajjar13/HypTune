import os, zipfile, tarfile,  argparse, yaml, re, warnings
from collections import Counter

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
    Embedding, LSTM, GRU, SimpleRNN, Attention, Conv2D, Flatten,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, TimeDistributed
    )
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_model_type(model, model_format, file_path):
    """
    Detects the type of machine learning model based on its architecture and output tensors.
    
    Supported model types:
    - Classification (CNN, MLP)
    - Object Detection
    - NLP/Transformer
    - RNN/LSTM/GRU
    - GAN
    - Autoencoder/VAE
    - Segmentation
    """
    model_type = "Unknown"
    
    if model_format == "keras":
        model_type = detect_keras_model_type(model)
    elif model_format == "pytorch":
        model_type = detect_pytorch_model_type(model)
    elif model_format == "tensorflow":
        model_type = detect_tensorflow_model_type(model)
    elif model_format == "onnx":
        model_type = detect_onnx_model_type(model)
    elif model_format == "tflite":
        model_type = detect_tflite_model_type(model)
        
    return model_type



def detect_keras_model_type(model):
    """
    Identify the type of Keras model based on its architecture.
    
    Supported types:
    - Text Classification
    - Text Generation
    - Machine Translation
    - Spell Checking
    - Image Classification
    - Object Detection
    """
    layers = model.layers
    output_shape = model.output_shape
    
    # Extract layer class names for more robust checking
    layer_class_names = [layer.__class__.__name__ for layer in layers]
    print(layer_class_names)    
    # Basic checks
    has_embedding = 'Embedding' in layer_class_names
    has_lstm = 'LSTM' in layer_class_names
    has_gru = 'GRU' in layer_class_names
    has_rnn = has_lstm or has_gru
    has_attention = 'Attention' in layer_class_names
    has_time_distributed = 'TimeDistributed' in layer_class_names
    
    # For image models
    has_conv = 'Conv2D' in layer_class_names
    has_flatten = 'Flatten' in layer_class_names
    has_pooling = ('GlobalAveragePooling2D' in layer_class_names) or ('GlobalMaxPooling2D' in layer_class_names)
    has_flat = has_flatten or has_pooling
    has_dense = 'Dense' in layer_class_names
    
    # Output characteristics
    is_sequence_output = isinstance(output_shape, tuple) and len(output_shape) == 3
    output_vocab_size = output_shape[-1] if isinstance(output_shape, tuple) else 0
    is_small_vocab = output_vocab_size < 100
    
    # Additional checks for TimeDistributed layers
    # Get TimeDistributed output vocabulary size
    td_output_size = 0
    for layer in layers:
        if isinstance(layer, TimeDistributed):
            if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                if isinstance(layer.output_shape, tuple) and len(layer.output_shape) >= 3:
                    td_output_size = layer.output_shape[-1]
    
    # TEXT MODELS IDENTIFICATION
    if has_embedding and has_rnn:
        # Spell Checking: LSTM/GRU + TimeDistributed with small vocabulary
        if has_time_distributed and td_output_size <= 100:
            return "Spell Checking"
        
        # Machine Translation: LSTM/GRU + TimeDistributed with large vocabulary
        if has_time_distributed and td_output_size > 100:
            return "Machine Translation"
        
        # Text Classification: LSTM/GRU without sequence output
        if not is_sequence_output or (output_shape[-1] < 100 and not has_time_distributed):
            return "Text Classification"
        
        # Text Generation: LSTM/GRU with sequence output
        if is_sequence_output:
            return "Text Generation"
    
    # IMAGE MODELS IDENTIFICATION
    if has_conv:
        if has_flat and has_dense:
            if isinstance(output_shape, tuple) and len(output_shape) == 2 and output_shape[-1] <= 1000:
                return "Image Classification"
        
        if isinstance(output_shape, tuple) and len(output_shape) >= 4 and output_shape[-1] >= 10:
            return "Object Detection"
    
    return "Unknown"

def detect_pytorch_model_type(model):
    def has_layer(model, layer_types):
        return any(isinstance(layer, layer_types) for layer in model.modules())

    # Text
    if has_layer(model, (nn.Embedding,)):
        if has_layer(model, (nn.LSTM, nn.GRU)):
            if has_layer(model, nn.MultiheadAttention):
                return "Machine Translation"
            # Attempt heuristic output shape detection
            for m in model.modules():
                if isinstance(m, nn.Linear) and m.out_features < 100:
                    return "Spell Checking"
            return "Text Generation"  # default if 3D output implied
        return "Text Classification"

    # Image
    if has_layer(model, nn.Conv2d):
        if has_layer(model, (nn.AdaptiveAvgPool2d, nn.Flatten)) and has_layer(model, nn.Linear):
            return "Image Classification"
        
        # Object Detection heuristics: large output, multiple heads, no flatten
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels >= 10 and m.kernel_size == (1, 1):
                return "Object Detection"
        return "Object Detection"

    return "Unknown"

def detect_tensorflow_model_type(model):
    try:
        # Get the default serving signature (used for inference)
        signature = model.signatures["serving_default"]

        # Extract structured output tensor names
        output_tensors = [tensor.name.lower() for tensor in signature.structured_outputs.values()]

        print("Output Tensors:", output_tensors)  # Debugging

        #  Object Detection should be checked **first**
        if any("box" in name or "detection" in name or "anchor" in name for name in output_tensors):
            return "Object Detection"

        #  NLP/Transformer check
        if any("token" in name or "embedding" in name or "attention" in name for name in output_tensors):
            return "Transformer/NLP"

        #  Classification check (Checked AFTER Object Detection to avoid false positives)
        if any("logits" in name or "probs" in name or "class" in name for name in output_tensors):
            return "Classification"

        #  Segmentation check
        if any("mask" in name or "segment" in name for name in output_tensors):
            return "Segmentation"

    except Exception as e:
        print(f"Error detecting TensorFlow model type: {e}")

    return "Unknown TensorFlow Model"

def detect_onnx_model_type(model):
    # Extract node types and output tensor names
    node_types = [node.op_type for node in model.graph.node]
    node_counter = Counter(node_types)
    output_names = [output.name.lower() for output in model.graph.output]
    
    # Check for NLP/Transformer
    if ('Attention' in node_counter or 'MatMul' in node_counter) and 'Embedding' in node_counter:
        return "Transformer/NLP"
    
    # Check for RNN/LSTM/GRU
    if any(x in node_counter for x in ['LSTM', 'GRU', 'RNN']):
        rnn_types = []
        if 'LSTM' in node_counter:
            rnn_types.append('LSTM')
        if 'GRU' in node_counter:
            rnn_types.append('GRU')
        if 'RNN' in node_counter and len(rnn_types) == 0:
            rnn_types.append('RNN')
        return f"Recurrent Network ({', '.join(rnn_types)})"
    
    # Check output tensor names for classification vs detection
    if any('logits' in name or 'probs' in name or 'class' in name for name in output_names):
        return "Classification"
    
    if any('box' in name or 'detection' in name for name in output_names):
        return "Object Detection"
    
    if any('mask' in name or 'segment' in name for name in output_names):
        return "Segmentation"
    
    # If model has many convolutions but doesn't fit other categories
    if node_counter.get('Conv', 0) > 3:
        return "Convolutional Neural Network (CNN)"
    
    return "Unknown ONNX Model"

def detect_tflite_model_type(interpreter):
    # Get output tensor details
    output_details = interpreter.get_output_details()
    output_names = [detail['name'].lower() for detail in output_details]
    
    # Examine input shapes for clues
    input_details = interpreter.get_input_details()
    
    # Check for NLP from input shape (batch, sequence_length)
    if len(input_details) > 0:
        input_shape = input_details[0]['shape']
        if len(input_shape) == 2:  # (batch, sequence_length) suggests NLP
            return "NLP/Sequence Model"
    
    # Check outputs for model type
    if any('logits' in name or 'prob' in name or 'class' in name for name in output_names):
        return "Classification"
    
    if any('box' in name or 'detection' in name for name in output_names):
        return "Object Detection"
    
    if any('mask' in name or 'segment' in name for name in output_names):
        return "Segmentation"
    
    # Get all tensor details for further analysis
    tensor_details = []
    for i in range(interpreter.get_tensor_details_size()):
        tensor_details.append(interpreter.get_tensor_details()[i])
    
    # Look for patterns in tensor names
    all_tensor_names = [detail['name'].lower() for detail in tensor_details]
    
    if any('lstm' in name or 'rnn' in name or 'gru' in name for name in all_tensor_names):
        rnn_types = []
        if any('lstm' in name for name in all_tensor_names):
            rnn_types.append('LSTM')
        if any('gru' in name for name in all_tensor_names):
            rnn_types.append('GRU')
        if len(rnn_types) == 0:
            rnn_types.append('RNN')
        return f"Recurrent Network ({', '.join(rnn_types)})"
    
    return "Unknown TFLite Model"

def load_model(file_path):
    """
    Load and identify the type of a machine learning model.
    
    Supported formats:
    - Keras (.h5)
    - TensorFlow SavedModel (.pb)
    - TFLite (.tflite)
    - PyTorch (.pt, .pth)
    - TorchScript (.torchscript)
    - ONNX (.onnx)
    
    Returns:
    - Model object
    - Model type
    """
    print(f"\n===== Loading model: {os.path.basename(file_path)} =====")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return None, "Unknown"
        
    ext = os.path.splitext(file_path)[-1].lower()
    model = None
    model_type = "Unknown"
    model_format = None
    
    try:
        # Keras .h5 model
        if ext == ".h5":
            model_format = "keras"
            try:
                model = tf.keras.models.load_model(file_path, compile = False)
                print(" Keras (.h5) Model Loaded Successfully!")
                model.summary()
                model_type = detect_model_type(model, model_format, file_path)
            except Exception as e:
                print(f"Error loading Keras model: {e}")

        # TensorFlow SavedModel
        elif ext == ".pb":
            model_format = "tensorflow"
            try:
                model_dir = os.path.dirname(file_path)
                model = tf.saved_model.load(model_dir)
                print(" TensorFlow SavedModel Loaded Successfully!")
                print("Model Signature Keys:", model.signatures.keys())
                model_type = detect_model_type(model, model_format, file_path)
            except Exception as e:
                print(f"Error loading TensorFlow SavedModel: {e}")

        # TFLite model
        elif ext == ".tflite":
            model_format = "tflite"
            try:
                interpreter = tf.lite.Interpreter(model_path=file_path)
                interpreter.allocate_tensors()
                print(" TFLite Model Loaded Successfully!")
                model = interpreter
                model_type = detect_model_type(model, model_format, file_path)
            except Exception as e:
                print(f"Error loading TFLite model: {e}")

        # PyTorch model
        elif ext in [".pt", ".pth"]:
            model_format = "pytorch"
            try:
                model = torch.load(file_path,weights_only=False, map_location=device)
                
                if isinstance(model, dict):
                    print(" This is a State Dictionary (weights only).")
                    print("Keys in state_dict:", list(model.keys())[:10], "..." if len(model.keys()) > 10 else "")
                    if "model" in model and isinstance(model["model"], nn.Module):
                        print("Found model architecture in checkpoint.")
                        model = model["model"]
                        model_type = detect_model_type(model, model_format, file_path)
                    else:
                        print("Cannot determine model type from weights only.")
                        model_type = "Unknown (weights only)"
                elif isinstance(model, nn.Module):
                    print("Full PyTorch Model Loaded Successfully!")
                    model_type = detect_model_type(model, model_format, file_path)
                else:
                    print(" Unrecognized PyTorch format.")
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")

        # TorchScript model
        elif ext == ".torchscript":
            model_format = "pytorch"
            try:
                model = torch.jit.load(file_path, map_location=device)
                print(" TorchScript Model Loaded Successfully!")
                model_type = detect_model_type(model, model_format, file_path)
            except Exception as e:
                print(f"Error loading TorchScript model: {e}")

        # ONNX model
        elif ext == ".onnx":
            model_format = "onnx"
            try:
                model = onnx.load(file_path)
                onnx.checker.check_model(model)
                print(" ONNX Model Loaded Successfully!")
                model_type = detect_model_type(model, model_format, file_path)
            except Exception as e:
                print(f"Error loading ONNX model: {e}")
        
        else:
            print(f"Unsupported file format: {ext}")
            return None, "Unknown"
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, "Unknown"
    
    # Print model type detection results
    print("\n===== Model Analysis =====")
    print(f"Format: {model_format.upper() if model_format else 'Unknown'}")
    print(f"Type: {model_type}")
    print("========================\n")
    
    return model, model_type


    #elif ext == ".ckpt":
    #    model = torch.load(file_path)  # Modify for TensorFlow
