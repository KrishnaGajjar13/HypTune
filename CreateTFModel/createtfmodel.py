from CreateTFModel.ImageClassification import ImageClassificationStudy
from CreateTFModel.ObjectDetection import ObjectDetectionStudy
from CreateTFModel.MachineTranslation import MachineTranslationStudy
from CreateTFModel.TextClassification import TextClassificationStudy

def CreateTFModel(Task: str, Hypmode : str ,size: str, imbalance: bool = False, class_imbalance: dict = None,data : list = [], labels = [], class_counts: dict = None,trials : int = 100):
    if Task == "image classification":
        ImageClassificationStudy(size,Hypmode ,data, labels, imbalance, class_imbalance, class_counts , trials)
    
    elif Task == "text classification":
        TextClassificationStudy(size,Hypmode ,data, labels, imbalance, class_imbalance, class_counts , trials) 
    
    elif Task == "object detection":
        ObjectDetectionStudy(size,Hypmode ,data, labels, imbalance, class_imbalance, class_counts , trials)

    elif Task == "machine translation":
        MachineTranslationStudy(size,Hypmode ,data, labels, imbalance, class_imbalance, class_counts , trials)
    
    elif Task == "spell checking":
        pass
    elif Task == "text generation":
        pass
    else: 
        raise ValueError(f"Unsupported task: {Task}. Supported tasks are: image classification, text classification, object detection, machine translation, spell checking, text generation.")

