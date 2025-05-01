# **HypTune**

**A Unified Framework for Hyperparameter Tuning in Machine Learning**

HypTune is a versatile framework designed to streamline hyperparameter tuning and training across multiple machine learning tasks and frameworks. With its configuration-driven approach, HypTune eliminates the need for repetitive code, enabling you to focus on experimentation and results.

---

## 🚀 **Getting Started**

### 1. Clone the Repository

```bash
git clone https://github.com/KrishnaGajjar13/HypTune.git
cd HypTune
```

### 2. Create a Configuration File

Define your experiment parameters in a YAML file. See the [Configuration Format](#-configuration-format) section for details.

### 3. Run Your Experiment

Execute the following command to start your experiment:

```bash
python HypTune/main.py --config path/to/your/config.yaml
```

---

## ⚙️ **Configuration Format**

The configuration file (`config.yaml`) defines the model, dataset, framework, task, and hyperparameter tuning settings. Below is an example:

```yaml
Model: "Path/to/model" # Optional: Path to a pre-trained model or weights
Dataset: "Path/to/data.jsonl" # Path to your dataset
Framework: "TensorFlow" # Options: TensorFlow | PyTorch
Task: "Text Classification" # Options: Image Classification | Object Detection | Text Classification | Machine Translation
Hypmode: "Full" # Options: Min | Moderate | Full
Trials: 100 # Number of trials for hyperparameter tuning

# Additional parameters for Machine Translation
InputLanguage: "en" # Source language (e.g., en, es, fr)
OutputLanguage: "fr" # Target language (e.g., en, es, fr)
```

---

## 🧠 **Supported Tasks and Models**

### **1. Image Classification**

- **Frameworks**: TensorFlow, PyTorch
- **Models**: MobileNet, ResNet, EfficientNet, DenseNet, Inception, Xception, and more.

### **2. Object Detection**

- **Frameworks**: TensorFlow, PyTorch
- **Models**: SSD, Faster R-CNN, RetinaNet, EfficientDet.

### **3. Text Classification**

- **Frameworks**: TensorFlow, PyTorch
- **Models**: RNN, LSTM, GRU, Transformers (e.g., BERT, RoBERTa, DistilBERT).

### **4. Machine Translation**

- **Frameworks**: PyTorch
- **Models**: Transformer-based multilingual architectures (e.g., MarianMT).

---

## 📂 **Dataset Format**

### **Text Classification**

Provide a JSONL file with labeled text samples:

```json
{"text": "I love this product!", "label": "positive"}
{"text": "The movie was okay.", "label": "neutral"}
{"text": "Worst service ever.", "label": "negative"}
```

### **Machine Translation**

Provide a JSONL file with input-output language pairs:

```json
{"input": "Hello, how are you?", "output": "Bonjour, comment ça va?"}
{"input": "The stars are beautiful tonight.", "output": "Les étoiles sont belles ce soir."}
{"input": "It is raining heavily today.", "output": "Il pleut abondamment aujourd'hui."}
```

### **Object Detection**

Provide COCO-style annotations with image paths and bounding boxes.

---

## 🔧 **Hyperparameter Tuning Modes**

HypTune offers three levels of hyperparameter tuning:

| **Mode**     | **Description**                                             |
| ------------ | ----------------------------------------------------------- |
| **Min**      | Minimal tuning for quick prototyping.                       |
| **Moderate** | Balanced tuning for a tradeoff between speed and accuracy.  |
| **Full**     | Exhaustive search for maximum control over hyperparameters. |

---

## 🧩 **Framework Support**

- **TensorFlow/Keras**: Fully supported for classification and object detection.
- **PyTorch**: Fully supported for classification, object detection, and machine translation.
- **Extendable**: Add support for custom frameworks or tasks using the modular structure.

---

## 🌟 **Features**

- **Configuration-Driven**: Define experiments in a YAML file.
- **Multi-Framework Support**: Seamlessly switch between TensorFlow and PyTorch.
- **Task-Specific Optimization**: Predefined pipelines for classification, detection, and translation.
- **Scalable Tuning**: Supports Optuna for efficient hyperparameter optimization.
- **Customizable**: Easily extend with new models, datasets, or tasks.

---

## **Planned Enhancements**

- [ ] Native Hugging Face integration for Transformers.
- [ ] Support for Vision Transformers and Large Language Models (LLMs).
- [ ] Web-based YAML editor and experiment dashboard.
- [ ] Integration with TensorBoard and Weights & Biases for tracking.

---

## **Contributing**

We welcome contributions! Feel free to open issues, suggest features, or submit pull requests.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
