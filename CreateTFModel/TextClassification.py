import os
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from transformers.keras_callbacks import KerasMetricCallback
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

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
# Model Building Functions
# -----------------------------
def build_non_transformer_model(trial, architecture, input_len, vocab_size, num_classes, is_binary):
    embedding_dim = trial.suggest_categorical("embedding_dim", [50, 100, 200])
    rnn_units = trial.suggest_int("rnn_units", 32, 256, step=32)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_len))

    if architecture == "SimpleRNN":
        model.add(SimpleRNN(rnn_units))
    elif architecture == "LSTM":
        model.add(LSTM(rnn_units))
    elif architecture == "GRU":
        model.add(GRU(rnn_units))
    elif architecture == "BidirectionalLSTM":
        model.add(Bidirectional(LSTM(rnn_units)))
    elif architecture == "BidirectionalGRU":
        model.add(Bidirectional(GRU(rnn_units)))
    elif architecture == "StackedLSTM":
        model.add(LSTM(rnn_units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(rnn_units))
    elif architecture == "StackedBidirectionalLSTM":
        model.add(Bidirectional(LSTM(rnn_units, return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(rnn_units)))
    elif architecture == "StackedBidirectionalGRU":
        model.add(Bidirectional(GRU(rnn_units, return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(GRU(rnn_units)))

    model.add(Dropout(dropout_rate))
    if is_binary:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(num_classes, activation="softmax"))

    return model


def build_transformer_model(trial, model_name, num_classes, fine_tune=True):
    # Create a custom Keras model that wraps the transformer
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    
    # Load the transformer model as a layer
    transformer_model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    
    # If not fine-tuning, freeze the transformer layers
    if not fine_tune:
        transformer_model.trainable = False
    
    # Call the transformer model with individual inputs, not a dictionary
    outputs = transformer_model(input_ids=input_ids, attention_mask=attention_mask, training=True)
    
    # Get the logits from the transformer output
    logits = outputs.logits
    
    # Create a Keras model with our inputs and outputs
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)
    
    return model

# -----------------------------
# Main Study Function
# -----------------------------
def TextClassificationStudy(
    size: str,
    Hypmode: str,
    texts,
    labels,
    imbalance: bool = False,
    class_imbalance: dict = None,
    class_counts: dict = None,  
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
    
    # Tokenize text
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
            
            # Process texts into transformer format
            encoded_data = transformer_tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=max_sequence_length, 
                return_tensors='tf'
            )
            
            # Create train and validation splits
            train_indices, val_indices = train_test_split(
                range(len(labels)), 
                test_size=0.2, 
                stratify=labels
            )
            
            # Extract input tensors
            input_ids_train = tf.gather(encoded_data['input_ids'], train_indices)
            attention_mask_train = tf.gather(encoded_data['attention_mask'], train_indices)
            input_ids_val = tf.gather(encoded_data['input_ids'], val_indices)
            attention_mask_val = tf.gather(encoded_data['attention_mask'], val_indices)
            
            y_train = tf.gather(labels, train_indices)
            y_val = tf.gather(labels, val_indices)
            
            # Build model using our new function
            model = build_transformer_model(trial, model_name, num_classes)
            
            learning_rate = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
            
            # Use appropriate loss function
            loss = "binary_crossentropy" if is_binary else "sparse_categorical_crossentropy"
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss,
                metrics=["accuracy"]
            )
            
            # Use standard Keras callbacks
            callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
            
            # Train model with separate inputs
            history = model.fit(
                [input_ids_train, attention_mask_train],  # Pass as list instead of dict
                y_train,
                validation_data=([input_ids_val, attention_mask_val], y_val),
                epochs=3,
                batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            _, val_acc = model.evaluate([input_ids_val, attention_mask_val], y_val, verbose=0)
            
        else:
            architecture = trial.suggest_categorical("architecture", architectures)
            val_split = trial.suggest_float("val_split", 0.1, 0.3)
            X_train, X_val, y_train, y_val = train_test_split(
                padded_sequences, labels, test_size=val_split, stratify=labels
            )

            model = build_non_transformer_model(
                trial, architecture, max_sequence_length, max_vocab_size, num_classes, is_binary
            )

            learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            loss = "binary_crossentropy" if is_binary else "sparse_categorical_crossentropy"
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss,
                metrics=["accuracy"]
            )

            callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                callbacks=callbacks,
                verbose=0
            )

            _, val_acc = model.evaluate(X_val, y_val, verbose=0)

        # Log trial results
        trial.set_user_attr("val_accuracy", val_acc)
        return val_acc

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