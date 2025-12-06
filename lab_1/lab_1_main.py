import os
import re
import numpy as np
from glob import glob
import random
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

USE_SPACY = False
if USE_SPACY:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

DATA_DIR = "data/aclImdb"
MAX_VOCAB = 20000
MAX_LEN = 200
EMBED_DIM = 100
BATCH_SIZE = 128
EPOCHS = 10


def read_imdb_split(split_dir):
    texts = []
    labels = []
    for label in ("pos", "neg"):
        files = glob(os.path.join(split_dir, label, "*.txt"))
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label == "pos" else 0)
    return texts, labels


def clean_text(s):
    s = s.lower()
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def optional_lemmatize(texts):
    if not USE_SPACY:
        return texts
    out = []
    for doc in nlp.pipe(texts, batch_size=128):
        tokens = [t.lemma_ for t in doc if not t.is_punct and not t.is_space]
        out.append(" ".join(tokens))
    return out


def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    X_train, y_train = read_imdb_split(train_dir)
    X_test, y_test = read_imdb_split(test_dir)
    return X_train, y_train, X_test, y_test


def build_model(vocab_size, maxlen):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=maxlen),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    print("Loading data...")
    X_train_raw, y_train, X_test_raw, y_test = load_data(DATA_DIR)
    print(f"Train samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

    X_train = [clean_text(t) for t in X_train_raw]
    X_test = [clean_text(t) for t in X_test_raw]

    X_train = optional_lemmatize(X_train)
    X_test = optional_lemmatize(X_test)

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    seq_train = tokenizer.texts_to_sequences(X_train)
    seq_test = tokenizer.texts_to_sequences(X_test)
    X_train_seq = pad_sequences(seq_train, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_seq = pad_sequences(seq_test, maxlen=MAX_LEN, padding='post', truncating='post')

    X_tr, X_val, y_tr, y_val = train_test_split(X_train_seq, np.array(y_train),
                                                test_size=0.1, random_state=42, stratify=y_train)

    model = build_model(MAX_VOCAB, MAX_LEN)
    model.summary()

    ckpt = ModelCheckpoint("best_imdb_cnn.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
    es = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1)

    history = model.fit(X_tr, y_tr,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[ckpt, es])

    print("Evaluating on test set...")
    loss, acc = model.evaluate(X_test_seq, np.array(y_test))
    print(f"Test accuracy: {acc:.4f}")

    y_pred_proba = model.predict(X_test_seq, batch_size=1024).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report)
    df_report.to_csv("classification_report.csv", index=True)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.matshow(cm, cmap='Blues', alpha=0.7)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['neg','pos']); ax.set_yticklabels(['neg','pos'])
    plt.title('Confusion matrix (test)')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("\n=== Example Predictions ===")
    indices = random.sample(range(len(X_test)), 5)
    for i in indices:
        text = X_test_raw[i]
        seq = tokenizer.texts_to_sequences([clean_text(text)])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        prob = model.predict(padded, verbose=0)[0][0]
        pred_label = "Positive" if prob >= 0.5 else "Negative"
        true_label = "Positive" if y_test[i] == 1 else "Negative"
        print(f"\n--- Review {i} ---")
        print(f"True Label: {true_label}")
        print(f"Predicted: {pred_label} (prob={prob:.4f})")
        print("Review text:")
        print(text[:400] + ("..." if len(text) > 400 else ""))  # truncate long texts


if __name__ == "__main__":
    main()
