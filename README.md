# **HypTune**  
**Framework for ML Hyperparameter Tuning & Training**

HypTune is a framework designed to simplify the process of hyperparameter tuning across various machine learning tasks and frameworks. Whether you're working on classification, object detection, or machine translation, HypTune offers a flexible, configuration-driven way to orchestrate experiments — no need to rewrite your training logic every time.

---

## 🚀 **How to Set Up**

1. **Clone the Repository**
   ```
   git clone https://github.com/KrishnaGajjar13/ParamForge.git
   cd ParamForge
   ```

2. **Create a YAML Configuration File**  
   Define your model path, dataset, framework, task, and hyperparameter tuning mode.

3. **Run from Terminal**
   ```
   python HypTune/main.py --config path/to/your/config.yaml
   ```

---

## ⚙️ **Configuration Format (config.yaml)**

```
Model: "Path/to/model"         # Optional: "Path/to/model", "Path/to/weights", or leave empty
Dataset: "Path/to/data.jsonl"
Framework: "TensorFlow"        # Options: TensorFlow | PyTorch
Task: "Text Classification"    # Options: Image Classification | Object Detection | Text Classification | Machine Translation
Hypmode: "Full"                # Options: Min | Moderate | Full
Trials: 100                    # Number of tuning trials (e.g., for Optuna)

# For Machine Translation only
InputLanguage: "en"            # Options: en, es, fr, de, it, pt, zh, ja, ko
OutputLanguage: "fr"           # Options: en, es, fr, de, it, pt, zh, ja, ko
```

---

## 🧠 **Supported Model Types**

- **Classification**  
  - ANN, CNN, RNN, LSTM, GRU

- **Object Detection**  
  - COCO-style detection models

- **Text Classification**  
  - RNN-based and Transformer-based

- **Machine Translation**  
  - Transformer-based multilingual architectures

---

## 📂 **Dataset Format Guidelines**

### **Text Classification**
JSONL file with labeled samples:
```
{"text": "I love this product!", "label": "positive"}
{"text": "The movie was okay.", "label": "neutral"}
{"text": "Worst service ever.", "label": "negative"}
```

### **Machine Translation**
JSONL file with input-output language pairs:
```
{"input": "Hello, how are you?", "output": "Bonjour, comment ça va?"}
{"input": "The stars are beautiful tonight.", "output": "Les étoiles sont belles ce soir."}
{"input": "It is raining heavily today.", "output": "Il pleut abondamment aujourd'hui."}
```

---

## 🔧 **Hyperparameter Modes**

HypTune supports **three modes** for hyperparameter tuning across supported model types and frameworks:

| Mode     | Description                              |
|----------|------------------------------------------|
| **Min**      | Minimal configuration — quick setup for rapid prototyping |
| **Moderate** | Balanced tuning — tradeoff between speed and accuracy |
| **Full**     | Exhaustive search — all tunable parameters for maximum control |

---

## 🧩 **Framework Support**

- ✅ **TensorFlow / Keras**
- ✅ **PyTorch**
- ⚙️ Extendable: Add your own framework logic using the modular structure.

---

## 📁 **Project Structure Overview**

```
HypTune/
├── main.py                   # Entry point
├── GetModel.py               # Model type detection logic
├── GetDataset.py             # Dataset input handler
├── confi.yaml                # Config 
├── CreateTFModel/            # Tuning logic per Keras/Tensorflow
│   ├── ImageClassification.py
│   ├── MachineTranslation.py
|   ├── ObjectDetection.py
|   ├── TextClassification.py
│   └── createtfmodel.py
├── CreateTorchModel/            # Tuning logic for Torch
│   ├── ImageClassification.py
│   ├── MachineTranslation.py
|   ├── ObjectDetection.py
|   ├── TextClassification.py
│   └── createtorchmodel.py
```

Each module has a single responsibility, making it easy to plug into different ML pipelines.

---

## **Planned Enhancements**

- [ ] Native Hugging Face support  
- [ ] Custom callback integrations (e.g., TensorBoard, Weights & Biases)  
- [ ] Web-based YAML editor + dashboard  
- [ ] Support for Vision Transformers and LLM fine-tuning

---

## **Contributing**

We welcome pull requests, feature suggestions, and bug reports. Please open an issue or submit a PR.

---

## **License**

This project is licensed under the MIT License.

---
