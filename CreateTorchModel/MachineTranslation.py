import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import warnings

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
warnings.filterwarnings("ignore")

# -----------------------------
# Data Augmentation Functions
# -----------------------------
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

def synonym_replacement(sentence, n=2):
    words = word_tokenize(sentence)
    new_words = words.copy()
    random.shuffle(words)
    num_replaced = 0
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            new_words = [random.choice(synonyms) if w == word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return " ".join(new_words)

def random_deletion(sentence, p=0.1):
    words = word_tokenize(sentence)
    if len(words) == 1:
        return sentence
    return " ".join([w for w in words if random.random() > p]) or random.choice(words)

def random_swap(sentence, n=2):
    words = word_tokenize(sentence)
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

def eda(sentence):
    return random.choice([synonym_replacement, random_deletion, random_swap])(sentence)

# -----------------------------
# PyTorch Dataset
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_texts(texts, labels, max_vocab_size=20000, max_len=200):
    # Create a word-to-index mapping
    word_to_idx = {"<PAD>": 0, "<OOV>": 1}
    word_counts = {}
    
    # Count word frequencies
    for text in texts:
        for word in word_tokenize(text.lower()):
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    # Sort words by frequency and add to vocab
    for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
        if len(word_to_idx) < max_vocab_size:
            word_to_idx[word] = len(word_to_idx)
    
    # Convert texts to sequences
    sequences = []
    for text in texts:
        seq = [word_to_idx.get(word.lower(), word_to_idx["<OOV>"]) for word in word_tokenize(text)]
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [word_to_idx["<PAD>"]] * (max_len - len(seq))
        sequences.append(seq)
    
    # Convert to PyTorch tensors
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    
    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return sequences_tensor, labels_tensor, word_to_idx, le

# -----------------------------
# PyTorch Model Classes
# -----------------------------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, rnn_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = output[:, -1, :]  # Take the last output
        output = self.dropout(output)
        output = self.fc(output)
        return output

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        output = hidden[-1]  # Take the last hidden state
        output = self.dropout(output)
        output = self.fc(output)
        return output

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, rnn_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        output = hidden[-1]  # Take the last hidden state
        output = self.dropout(output)
        output = self.fc(output)
        return output

class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(BidirectionalLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units * 2, num_classes)  # *2 because bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        # Concatenate the last hidden state from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout(hidden)
        output = self.fc(output)
        return output

class BidirectionalGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(BidirectionalGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, rnn_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units * 2, num_classes)  # *2 because bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        # Concatenate the last hidden state from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout(hidden)
        output = self.fc(output)
        return output

class StackedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(StackedLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(rnn_units, rnn_units, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (_, _) = self.lstm1(embedded)
        output = self.dropout1(output)
        output, (hidden, _) = self.lstm2(output)
        output = hidden[-1]  # Take the last hidden state
        output = self.dropout2(output)
        output = self.fc(output)
        return output

class StackedBidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(StackedBidirectionalLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, rnn_units, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(rnn_units * 2, rnn_units, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (_, _) = self.lstm1(embedded)
        output = self.dropout1(output)
        output, (hidden, _) = self.lstm2(output)
        # Concatenate the last hidden state from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout2(hidden)
        output = self.fc(output)
        return output

class StackedBidirectionalGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(StackedBidirectionalGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, rnn_units, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.gru2 = nn.GRU(rnn_units * 2, rnn_units, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru1(embedded)
        output = self.dropout1(output)
        output, hidden = self.gru2(output)
        # Concatenate the last hidden state from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout2(hidden)
        output = self.fc(output)
        return output

# -----------------------------
# Transformer Model Function
# -----------------------------
def load_translation_model(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Translate sentences using transformer model
def translate_sentences(sentences, model, tokenizer):
    translated = []
    for sentence in sentences:
        # Tokenize the sentence
        tokens = tokenizer.encode(sentence, return_tensors="pt", padding=True, truncation=True)
        
        # Generate translation (output is a tensor)
        with torch.no_grad():
            translation = model.generate(tokens)
        
        # Decode the translation back to text
        translated_sentence = tokenizer.decode(translation[0], skip_special_tokens=True)
        translated.append(translated_sentence)
    return translated

# -----------------------------
# Training Loop
# -----------------------------
def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
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
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load the best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

# -----------------------------
# Model Evaluation
# -----------------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

# -----------------------------
# Main Study Function for Translation
# -----------------------------
def MachineTranslationStudy(
    size: str,
    Hypmode: str,
    texts,
    labels,
    num_trials: int = 10,
    log_csv_path: str = "translation_trials_log.csv"
):
    max_vocab_size = 20000
    max_sequence_length = 200
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Preprocess data
    sequences_tensor, labels_tensor, word_to_idx, label_encoder = preprocess_texts(texts, labels, max_vocab_size, max_sequence_length)
    vocab_size = len(word_to_idx)
    num_classes = len(label_encoder.classes_)
    
    # Model pool based on dataset size
    model_pool = {
        "small": ["SimpleRNN", "LSTM", "GRU"],
        "medium": ["BidirectionalLSTM", "StackedLSTM", "BidirectionalGRU"],
        "large": ["StackedBidirectionalLSTM", "StackedBidirectionalGRU"]
    }
    
    if size in ["too small", "small"]:
        architectures = model_pool["small"]
    elif size == "medium":
        architectures = model_pool["medium"]
    else:
        architectures = model_pool["large"]
    
    # Initialize results dataframe
    columns = ["trial_number", "architecture", "embedding_dim", "rnn_units", "dropout", 
               "lr", "val_split", "batch_size", "val_accuracy"]
    results_df = pd.DataFrame(columns=columns)
    
    for trial in range(num_trials):
        for architecture in architectures:
            # Hyperparameter selection
            embedding_dim = random.choice([50, 100, 200])
            rnn_units = random.randint(32, 256)
            dropout_rate = random.uniform(0.1, 0.5)
            learning_rate = random.choice([0.001, 0.01, 0.0001])
            batch_size = random.choice([32, 64, 128])
            
            val_split = 0.2
            if size in ["medium", "large"] and Hypmode == "full":
                val_split = random.choice([0.15, 0.25, 0.35])
            
            # Split the data
            X_train, X_val, y_train, y_val = train_test_split(
                sequences_tensor, labels_tensor, test_size=val_split, 
                stratify=labels_tensor, random_state=42
            )
            
            # Create data loaders
            train_dataset = TextDataset(X_train, y_train)
            val_dataset = TextDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model based on architecture
            if architecture == "SimpleRNN":
                model = SimpleRNN(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
            elif architecture == "LSTM":
                model = LSTMModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
            elif architecture == "GRU":
                model = GRUModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
            elif architecture == "BidirectionalLSTM":
                model = BidirectionalLSTM(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
            elif architecture == "BidirectionalGRU":
                model = BidirectionalGRU(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
            elif architecture == "StackedLSTM":
                model = StackedLSTM(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
            elif architecture == "StackedBidirectionalLSTM":
                model = StackedBidirectionalLSTM(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
            elif architecture == "StackedBidirectionalGRU":
                model = StackedBidirectionalGRU(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
            
            model.to(device)

            # Set number of epochs
            epochs = 10 if Hypmode in ["min", "moderate"] else 20
            
            # Train the model
            _, val_acc = train_model(model, train_loader, val_loader, epochs, learning_rate, device)
            
            # Log results
            result = {
                "trial_number": trial,
                "architecture": architecture,
                "embedding_dim": embedding_dim,
                "rnn_units": rnn_units,
                "dropout": dropout_rate,
                "lr": learning_rate,
                "val_split": val_split,
                "batch_size": batch_size,
                "val_accuracy": val_acc
            }
            
            results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
            
            # Save results to CSV
            results_df.to_csv(log_csv_path, index=False)
            
            print(f"Trial {trial}, Architecture: {architecture}, Validation Accuracy: {val_acc:.4f}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Example parallel dataset
    source_texts = [
        "Hello, how are you?",
        "I love machine learning.",
        "The weather is nice today.",
        "What is your name?",
        "This is a test sentence."
    ] * 100  # Simulate dataset size
    
    target_texts = [
        "Bonjour, comment ça va ?",
        "J'adore l'apprentissage automatique.",
        "Il fait beau aujourd'hui.",
        "Quel est ton nom ?",
        "Ceci est une phrase de test."
    ] * 100  # Simulated French translations

    # Combine to form a single dataset with labels for language
    all_texts = source_texts + target_texts
    all_labels = ["en"] * len(source_texts) + ["fr"] * len(target_texts)
    
    # Determine dataset size category
    avg_len = sum(len(t.split()) for t in all_texts) / len(all_texts)
    if avg_len <= 10:
        dataset_size = "small"
    elif avg_len <= 20:
        dataset_size = "medium"
    else:
        dataset_size = "large"
    
    print(f"Dataset size category: {dataset_size}")
    
    # Run study
    MachineTranslationStudy(
        size=dataset_size,
        Hypmode="full",
        texts=all_texts,
        labels=all_labels,
        num_trials=3,  # Reduced for demonstration
        log_csv_path="translation_trials_log.csv"
    )
    
    # Example of using transformers for translation
    print("\nExample of transformer-based translation:")
    model, tokenizer = load_translation_model("en", "fr")
    translated = translate_sentences(source_texts[:3], model, tokenizer)
    
    for src, tgt in zip(source_texts[:3], translated):
        print(f"Source: {src}")
        print(f"Translation: {tgt}")
        print()

if __name__ == "__main__":
    main()