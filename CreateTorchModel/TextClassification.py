import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from collections import Counter

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
warnings.filterwarnings("ignore")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# Preprocessing Functions
# -----------------------------
class Tokenizer:
    def __init__(self, num_words=None, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()
        self.word_index[oov_token] = 1  # Reserve 0 for padding
        self.index_word[1] = oov_token
        
    def fit_on_texts(self, texts):
        for text in texts:
            for word in word_tokenize(text.lower()):
                self.word_counts[word] += 1
                
        # Sort words by frequency
        vocab = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        if self.num_words:
            vocab = vocab[:self.num_words-2]  # -2 for padding and OOV
            
        # Create word index
        for i, (word, _) in enumerate(vocab):
            idx = i + 2  # Reserve 0 for padding, 1 for OOV
            self.word_index[word] = idx
            self.index_word[idx] = word
            
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in word_tokenize(text.lower()):
                if word in self.word_index:
                    seq.append(self.word_index[word])
                else:
                    seq.append(1)  # OOV token
            sequences.append(seq)
        return sequences

def pad_sequences(sequences, maxlen=None, padding='post', truncating='post'):
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
        
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'post':
                new_seq = seq[:maxlen]
            else:  # pre
                new_seq = seq[len(seq)-maxlen:]
        else:
            new_seq = seq.copy()
            
        if len(new_seq) < maxlen:
            pad_len = maxlen - len(new_seq)
            if padding == 'post':
                new_seq.extend([0] * pad_len)
            else:  # pre
                new_seq = [0] * pad_len + new_seq
        
        padded.append(new_seq)
    
    return torch.tensor(padded, dtype=torch.long)

def preprocess_texts(texts, labels, max_vocab_size=20000, max_len=200):
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return padded, torch.tensor(labels, dtype=torch.long), tokenizer, le

# -----------------------------
# Model Architectures
# -----------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, rnn_units, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        # Take the output of the last time step
        output = output[:, -1, :]
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
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        output = output[:, -1, :]
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
        x = self.embedding(x)
        output, _ = self.gru(x)
        # Take the output of the last time step
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output

class BidirectionalLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(BidirectionalLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units * 2, num_classes)  # * 2 for bidirectional
        
    def forward(self, x):
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)
        # Take the output of the last time step from both directions
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output

class BidirectionalGRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(BidirectionalGRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, rnn_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units * 2, num_classes)  # * 2 for bidirectional
        
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.gru(x)
        # Take the output of the last time step from both directions
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output

class StackedLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(StackedLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, rnn_units, batch_first=True, return_sequences=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(rnn_units, rnn_units, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm1(x)
        output = self.dropout1(output)
        output, _ = self.lstm2(output)
        # Take the output of the last time step
        output = output[:, -1, :]
        output = self.dropout2(output)
        output = self.fc(output)
        return output

class StackedBidirectionalLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(StackedBidirectionalLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, rnn_units, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(rnn_units * 2, rnn_units, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units * 2, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm1(x)
        output = self.dropout1(output)
        output, _ = self.lstm2(output)
        # Take the output of the last time step
        output = output[:, -1, :]
        output = self.dropout2(output)
        output = self.fc(output)
        return output

class StackedBidirectionalGRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(StackedBidirectionalGRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, rnn_units, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.gru2 = nn.GRU(rnn_units * 2, rnn_units, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units * 2, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.gru1(x)
        output = self.dropout1(output)
        output, _ = self.gru2(output)
        # Take the output of the last time step
        output = output[:, -1, :]
        output = self.dropout2(output)
        output = self.fc(output)
        return output

# Fix StackedLSTM forward to handle return_sequences correctly
class FixedStackedLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate=0.2):
        super(FixedStackedLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(rnn_units, rnn_units, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rnn_units, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm1(x)
        output = self.dropout1(output)
        output, (h_n, c_n) = self.lstm2(output)
        # Take the output of the last time step
        output = output[:, -1, :]
        output = self.dropout2(output)
        output = self.fc(output)
        return output

# -----------------------------
# Training Functions
# -----------------------------
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
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
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
    
    val_loss = running_loss / len(dataloader)
    val_acc = correct / total
    return val_loss, val_acc

def build_non_transformer_model(trial, architecture, vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate):
    if architecture == "SimpleRNN":
        return SimpleRNNModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
    elif architecture == "LSTM":
        return LSTMModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
    elif architecture == "GRU":
        return GRUModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
    elif architecture == "BidirectionalLSTM":
        return BidirectionalLSTMModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
    elif architecture == "BidirectionalGRU":
        return BidirectionalGRUModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
    elif architecture == "StackedLSTM":
        return FixedStackedLSTMModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
    elif architecture == "StackedBidirectionalLSTM":
        return StackedBidirectionalLSTMModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
    elif architecture == "StackedBidirectionalGRU":
        return StackedBidirectionalGRUModel(vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

# Transformer model building function
class CustomTransformerModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomTransformerModel, self).__init__()
        self.transformer = base_model
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token output
        logits = self.classifier(pooled_output)
        return logits

def build_transformer_model(trial, model_name, num_classes, fine_tune=True):
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    
    if not fine_tune:
        for param in base_model.parameters():
            param.requires_grad = False
            
    return base_model

# -----------------------------
# Early Stopping Implementation
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            
        return self.early_stop
    
    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)

# -----------------------------
# Main Study Function
# -----------------------------
def TextClassificationStudy(
    size: str,
    Hypmode: str,
    texts,
    labels,
    num_trials: int = 10,
    log_csv_path: str = "text_trials_log.csv"
):
    max_vocab_size = 20000
    max_sequence_length = 200

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(labels))
    is_binary = num_classes == 2
    
    # Tokenize text for non-transformer models
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post", truncating="post")

    model_pool = {
        "small": ["SimpleRNN", "LSTM", "GRU"],
        "medium": ["BidirectionalLSTM", "StackedLSTM", "BidirectionalGRU"],
        "large": ["StackedBidirectionalLSTM", "StackedBidirectionalGRU"]
    }
    
    if size in ["too small", "small"]:
        architectures = model_pool["small"]
        use_transformer = False
    elif size == "medium":
        architectures = model_pool["medium"]
        use_transformer = True
    else:
        architectures = model_pool["large"]
        use_transformer = True

    def objective(trial):
        if use_transformer:
            model_name = trial.suggest_categorical("transformer_model", [
                "bert-base-uncased", 
                "distilbert-base-uncased", 
                "roberta-base"
            ])
            transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            encoded_texts = transformer_tokenizer(texts, padding=True, truncation=True, max_length=max_sequence_length, return_tensors='pt')
            
            input_ids = encoded_texts['input_ids']
            attention_mask = encoded_texts['attention_mask']
            
            # Split data
            val_split = trial.suggest_float("val_split", 0.1, 0.3)
            X_train_ids, X_val_ids, X_train_mask, X_val_mask, y_train, y_val = train_test_split(
                input_ids, attention_mask, torch.tensor(labels), test_size=val_split, stratify=labels
            )
            
            # Create datasets and dataloaders
            train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train)
            val_dataset = TensorDataset(X_val_ids, X_val_mask, y_val)
            
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Build and prepare model
            model = build_transformer_model(trial, model_name, num_classes)
            model = model.to(device)
            
            learning_rate = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Loss function
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_val_acc = 0
            epochs = 3
            
            for epoch in range(epochs):
                model.train()
                for batch in train_loader:
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    if isinstance(outputs, tuple):
                        loss = outputs[0]
                        logits = outputs[1]
                    else:
                        logits = outputs.logits
                        loss = criterion(logits, labels)
                    
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = [b.to(device) for b in batch]
                        
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        if isinstance(outputs, tuple):
                            logits = outputs[1]
                        else:
                            logits = outputs.logits
                            
                        _, predicted = torch.max(logits, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                val_acc = correct / total
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
        
        else:  # Non-transformer models
            architecture = trial.suggest_categorical("architecture", architectures)
            embedding_dim = trial.suggest_categorical("embedding_dim", [50, 100, 200])
            rnn_units = trial.suggest_int("rnn_units", 32, 256, step=32)
            dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
            
            # Split data
            val_split = trial.suggest_float("val_split", 0.1, 0.3)
            X_train, X_val, y_train, y_val = train_test_split(
                padded_sequences, torch.tensor(labels), test_size=val_split, stratify=labels
            )
            
            # Create dataset and dataloaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Build model
            model = build_non_transformer_model(
                trial, architecture, max_vocab_size, embedding_dim, rnn_units, num_classes, dropout_rate
            )
            model = model.to(device)
            
            # Optimizer and loss
            learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            criterion = nn.BCEWithLogitsLoss() if is_binary and num_classes == 2 else nn.CrossEntropyLoss()
            
            # Early stopping
            early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
            
            # Training loop
            epochs = 10
            best_val_acc = 0
            
            for epoch in range(epochs):
                # Train
                model.train()
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                
                # Validate
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                
                # Check early stopping
                if early_stopping(val_loss, model):
                    early_stopping.restore_weights(model)
                    break
                
                best_val_acc = max(best_val_acc, val_acc)
        
        # Log trial results
        trial.set_user_attr("val_accuracy", best_val_acc)
        return best_val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)

    # Save results
    trial_results = [{
        "trial_number": t.number,
        **t.params,
        "val_accuracy": t.user_attrs.get("val_accuracy")
    } for t in study.trials]

    df = pd.DataFrame(trial_results)
    df.to_csv(log_csv_path, index=False)
    print("Study complete. Best trial:")
    print(study.best_trial)

def main():
    # Example small dataset
    texts = ["This app is great", "Terrible product", "I love it", "Awful experience", "Fantastic service"] * 200
    labels = ["pos", "neg", "pos", "neg", "pos"] * 200

    TextClassificationStudy(
        size="medium",
        Hypmode="full",
        texts=texts,
        labels=labels,
        num_trials=10
    )

if __name__ == "__main__":
    main()