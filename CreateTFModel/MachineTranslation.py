import numpy as np
import pandas as pd
import random, optuna
import torch, os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional # type: ignore
from transformers import MarianMTModel, MarianTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
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
# Preprocessing Function
# -----------------------------
def preprocess_texts(texts, labels, max_vocab_size=20000, max_len=200):
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return padded, labels, tokenizer, le

# -----------------------------
# Non-Transformer Model Functions
# -----------------------------
def build_non_transformer_model(trial, architecture, input_len, vocab_size, num_classes):
    embedding_dim = trial.suggest_categorical("embedding_dim", [50, 100, 200])
    rnn_units = trial.suggest_int("rnn_units", 32, 256, step=32)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_len))

    if architecture == "SimpleRNN":
        model.add(SimpleRNN(rnn_units, activation="tanh"))
    elif architecture == "LSTM":
        model.add(LSTM(rnn_units, activation="tanh", recurrent_activation="sigmoid", use_bias=True, unroll=False))
    elif architecture == "GRU":
        model.add(GRU(rnn_units, activation="tanh", recurrent_activation="sigmoid", use_bias=True, reset_after=True))
    elif architecture == "BidirectionalLSTM":
        model.add(Bidirectional(LSTM(rnn_units, activation="tanh", recurrent_activation="sigmoid", use_bias=True, unroll=False)))
    elif architecture == "BidirectionalGRU":
        model.add(Bidirectional(GRU(rnn_units, activation="tanh", recurrent_activation="sigmoid", use_bias=True, reset_after=True)))
    elif architecture == "StackedLSTM":
        model.add(LSTM(rnn_units, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", use_bias=True, unroll=False))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(rnn_units, activation="tanh", recurrent_activation="sigmoid", use_bias=True, unroll=False))
    elif architecture == "StackedBidirectionalLSTM":
        model.add(Bidirectional(LSTM(rnn_units, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", use_bias=True, unroll=False)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(rnn_units, activation="tanh", recurrent_activation="sigmoid", use_bias=True, unroll=False)))
    elif architecture == "StackedBidirectionalGRU":
        model.add(Bidirectional(GRU(rnn_units, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", use_bias=True, reset_after=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(GRU(rnn_units, activation="tanh", recurrent_activation="sigmoid", use_bias=True, reset_after=True)))

    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax"))

    return model

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
# Main Study Function for Translation
# -----------------------------
def MachineTranslationStudy(
    size: str,
    Hypmode: str,
    texts,
    labels,
    imbalance: bool = False,
    class_imbalance: dict = None,
    class_counts: dict = None,
    num_trials: int = 10,
    log_csv_path: str = "translation_trials_log.csv"
):
    max_vocab_size = 20000
    max_sequence_length = 200

    # Tokenize text
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post", truncating="post")

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(labels))

    # Split data without stratification
    val_split = 0.2
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, labels, test_size=val_split, random_state=42
    )

    # Log dataset imbalance information if provided
    if imbalance:
        print("Dataset is imbalanced.")
        if class_imbalance:
            print(f"Class imbalance details: {class_imbalance}")
        if class_counts:
            print(f"Class counts: {class_counts}")

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

    def objective(trial):
        architecture = trial.suggest_categorical("architecture", architectures)
        val_split = 0.2
        if size in ["medium", "large"] and Hypmode == "full":
            val_split = trial.suggest_categorical("val_split", [0.15, 0.25, 0.35])

        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, labels, test_size=val_split, random_state=42
        )

        # Non-transformer models
        model = build_non_transformer_model(trial, architecture, max_sequence_length, max_vocab_size, num_classes)

        # Compile the model
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        epochs = 10 if Hypmode in ["min", "moderate"] else 20
        callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            verbose=0,
            callbacks=callbacks
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

        log_dict = {
            "trial_number": trial.number,
            "architecture": architecture,
            "embedding_dim": trial.params.get("embedding_dim"),
            "rnn_units": trial.params.get("rnn_units"),
            "dropout": trial.params.get("dropout"),
            "learning_rate": learning_rate,
            "val_split": val_split,
            "batch_size": trial.params.get("batch_size"),
            "val_accuracy": val_acc
        }

        if os.path.exists(log_csv_path):
            df = pd.read_csv(log_csv_path)
            df = pd.concat([df, pd.DataFrame([log_dict])], ignore_index=True)
        else:
            df = pd.DataFrame([log_dict])
        df.to_csv(log_csv_path, index=False)

        return val_acc

    # Create and optimize the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)

    print("Study complete. Best trial:")
    print(study.best_trial)

def main():
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

    # Determine dataset size category
    avg_len = sum(len(t.split()) for t in source_texts) / len(source_texts)
    if avg_len <= 10:
        dataset_size = "small"
    elif avg_len <= 20:
        dataset_size = "medium"
    else:
        dataset_size = "large"

    # Run study based on dataset size
    MachineTranslationStudy(
        size=dataset_size,
        Hypmode="full",
        texts=source_texts,
        labels=target_texts,
        num_trials=10,
        log_csv_path="translation_trials_log.csv"
    )

if __name__ == "__main__":
    main()
