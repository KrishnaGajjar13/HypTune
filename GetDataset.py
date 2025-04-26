import torch, warnings, json
import numpy as np
import pandas as pd
import os, zipfile, tarfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import torch.nn as nn
import torchvision.models as models 
import torchaudio.models as audiomodels
from collections import defaultdict

warnings.filterwarnings("ignore")

def ReadData(path : str, modeltype : str):

    """
    Reads a dataset from a given path and returns the data and labels.
    
    Args:
        path (str): Path to the dataset.
        modeltype (str): Type of model to be used.
    Returns:
        tuple: A tuple containing the data and labels.
    """

    if modeltype == 'image classification':
        # Assuming the dataset is in a directory structure where each subdirectory is a class
        data = []
        labels = []
        class_counts = defaultdict(int)  # To keep track of number of images per class

        for label in os.listdir(path):
            class_dir = os.path.join(path, label)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file)
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Case-insensitive check
                        data.append(file_path)
                        labels.append(label)
                        class_counts[label] += 1  # Increment class count

        return data, labels, dict(class_counts)
    
    elif modeltype == 'object detection':
        json_path = os.path.join(path, 'label.json')
        images_dir = os.path.join(path, 'images')

        with open(json_path, 'r') as f:
            label = json.load(f)

        # Build lookup tables
        id_to_filename = {img['id']: img['file_name'] for img in label['images']}
        id_to_annotations = defaultdict(list)
        class_counts = defaultdict(int)
        category_id_to_name = {cat['id']: cat['name'] for cat in label['categories']}

        for ann in label['annotations']:
            id_to_annotations[ann['image_id']].append(ann)
            class_counts[category_id_to_name[ann['category_id']]] += 1

        data = []
        labels = []

        for image_id, file_name in id_to_filename.items():
            image_path = os.path.join(images_dir, file_name)
            if os.path.exists(image_path):
                data.append(image_path)
                bboxes = []
                for ann in id_to_annotations[image_id]:
                    bbox = ann['bbox']  # [x, y, width, height]
                    label = category_id_to_name[ann['category_id']]
                    bboxes.append({'bbox': bbox, 'label': label})
                labels.append(bboxes)

        return data, labels, dict(class_counts)
    
    elif modeltype == 'text classification':
        data = []
        labels = []
        stats = defaultdict(int)  # holds both label counts and total token count
        total_input_tokens = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                text = sample['text']
                data.append(text)

                total_input_tokens += len(text.split())

                # Handle both single-label and multi-label
                if 'label' in sample:
                    label = sample['label']
                    labels.append(label)
                    stats[label] += 1

                elif 'labels' in sample:
                    label = sample['labels']
                    labels.append(label)
                    for l in label:
                        stats[l] += 1

                else:
                    raise ValueError("Missing 'label' or 'labels' field in line: " + line)

        stats['__total_tokens__'] = total_input_tokens
        return data, labels, dict(stats)

    
    elif modeltype in ['text generation', 'machine translation', 'spell checking']:
        inputs = []
        targets = []
        total_input_tokens = 0
        total_target_tokens = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())

                if 'input' in item and 'output' in item:
                    input_text = item['input']
                    output_text = item['output']
                    inputs.append(input_text)
                    targets.append(output_text)

                    total_input_tokens += len(input_text.split())
                    total_target_tokens += len(output_text.split())

                elif 'text' in item:
                    text = item['text']
                    inputs.append(text)
                    targets.append(text)

                    total_input_tokens += len(text.split())
                    total_target_tokens += len(text.split())

                else:
                    raise ValueError("Expected keys: 'input' & 'output' or 'text'")

        stats = {
            "__total_sentences__": len(inputs),
            "__total_tokens__": total_input_tokens + total_target_tokens
        }

        return inputs, targets, stats

    else:
        raise ValueError(f"Unsupported path type: {path}")


def CheckImbalance(count):
    """
    Checks if the dataset is imbalanced based on the count of samples in each class.
    
    Args:
        count (dict): Number of samples in the dataset.
    Returns:
        size : (str) Size of the dataset based on average samples per class.
        imbalance: True if the dataset is imbalanced, False otherwise.
        class_imbalance: (dict) Dictionary containing the imbalance ratio for each class.
    """
    average = sum(count.values()) / len(count)
    if average < 50:
        size = 'Insufficient'
    if average < 100:
        size = 'Too Small'
    elif average < 300:
        size = 'Small'
    elif average < 1000:
        size = 'Medium'
    elif average < 10000:
        size = 'Large'
    else:
        raise ValueError(f"Error reading dataset")
    imbalance = False
    class_imbalance = {}
    for key,value in count.items():
        if value/average < 0.8 or value/average > 1.2:
            margin = value/average
            print(average, margin)
            imbalance = True
            class_imbalance[key] = (margin-1) * 100
    return size, imbalance, class_imbalance


def DatasetCategory(modeltype, count):
    """
    Categorizes the dataset based on the modeltype, and it's size and returns a recommended size of model for trainning.    
    modeltype (str): Type of model to be used.
    count (dict): Number of samples in the dataset.
    """
    if modeltype == 'image classification':
        return CheckImbalance(count)

    elif modeltype == 'object detection':
        return CheckImbalance(count)

    elif modeltype == 'text classification':
        average =  (sum(count.values()) - count['__total_tokens__']) / (len(count) -1)
        if average < 300:
            size = 'Insufficient'
        if average <= 500:
            size =  'Too Small'
        elif average <= 1000:
            size =  'Small'
        elif average <= 5000:
            size = 'Medium'
        elif average <= 10000:
            size = 'Large'
        elif average > 10000:
            size = 'Out of Bounds'
        else:
            raise ValueError(f"Error reading dataset")
        
        class_imbalance = {}
        imbalance = False
        for key, value in count.items():
            if key != '__total_tokens__':
                if value/average < 0.75 or value/average > 1.25:
                    margin = value/average
                    imbalance = True
                    class_imbalance[key] =  (margin - 1) * 100 
        return size, imbalance, class_imbalance

    elif modeltype in ['Text Generation', 'machine translation', 'Spell Checking']:
        # Check if the dataset is imbalanced
        if count['__total_sentences__'] < 2000:
            size = 'Insufficient'
        elif count['__total_sentences__'] < 5000:
            size = 'Too Small'
        elif count['__total_sentences__'] >= 5000 and count['__total_sentences__'] <= 10000:
            size = 'Small'
        elif count['__total_sentences__'] <= 30000 and count['__total_tokens__'] > 10000:
            size = 'Medium'
        elif count['__total_sentences__'] <= 100000 and count['__total_tokens__'] > 30000:
            size = 'Large'
        imbalance = False
        class_imbalance = {}
        return size, imbalance, class_imbalance
    else:
        raise ValueError(f"Error reading dataset")
    
# Example usage
"""path = "/home/gajjar/Desktop/Sem VIII/Code/label.jsonl"
modeltype = "Object Detection"
#data, labels, count = read_dataset(path, modeltype)
count = {
  'class_0': 84,
  'class_1': 92,
  'class_2': 77,
  'class_3': 101,
  'class_4': 89
}
size, imbalance, class_imbalance = DatasetCategory(modeltype, count)
print(f"Count: {count}")
print(f"Dataset Size: {size}")
print(f"Imbalance: {imbalance}")
print(f"Class Imbalance: {class_imbalance}")"""
