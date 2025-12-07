import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


DATA_DIR = "data"
MODEL_PATH = "ner_bilstm_model.keras"

EMBED_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 8
MAX_LEN = 50


def load_conll_file(path):
    sentences, labels = [], []
    words, tags = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                parts = line.split()

                word = parts[0]

                tag = parts[-1]

                words.append(word)
                tags.append(tag)

    return sentences, labels


def load_data():
    x_train, y_train = load_conll_file(os.path.join(DATA_DIR, "train.txt"))
    x_val, y_val = load_conll_file(os.path.join(DATA_DIR, "valid.txt"))
    x_test, y_test = load_conll_file(os.path.join(DATA_DIR, "test.txt"))

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_vocab(sequences):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent in sequences:
        for word in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def build_tag_map(sequences):
    tags = {"PAD": 0}
    for sent in sequences:
        for tag in sent:
            if tag not in tags:
                tags[tag] = len(tags)
    return tags


def encode_and_pad(seqs, mapper, max_len):
    encoded = []
    for s in seqs:
        ids = [mapper.get(x, mapper.get("<UNK>", 0)) for x in s]
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        encoded.append(ids)
    return np.array(encoded)


def build_model(vocab_size, tag_count):
    model = keras.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True)),
        layers.Dense(tag_count, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def evaluate_model(model, x_test, y_test, tag_map):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=-1)

    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    mask = (y_test_flat != 0)

    inv_tag_map = {v: k for k, v in tag_map.items()}

    y_test_labels = [inv_tag_map[int(i)] for i in y_test_flat[mask]]
    y_pred_labels = [inv_tag_map[int(i)] for i in y_pred_flat[mask]]

    print("\nClassification Report (token-level, PAD excluded):")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=list(inv_tag_map.values()))
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest')
    plt.title("NER Confusion Matrix (token-level)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    masked_acc = np.mean(np.array(y_test_labels) == np.array(y_pred_labels))
    print(f"\nMasked token-level accuracy (PAD excluded): {masked_acc:.4f}")


def show_examples(model, x_test, x_test_text, y_test, tag_map, num_examples=5):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=-1)

    inv_tag_map = {v: k for k, v in tag_map.items()}

    indices = random.sample(range(len(x_test)), num_examples)

    print("\n🔹 Example NER Predictions:\n")
    for idx in indices:
        tokens = x_test_text[idx][:MAX_LEN]
        true_tags = [inv_tag_map[i] for i in y_test[idx][:MAX_LEN] if i != 0]
        pred_tags = [inv_tag_map[i] for i in y_pred[idx][:MAX_LEN] if i != 0]

        print(f"Sentence {idx+1}:")
        for t, pt, tt in zip(tokens, pred_tags, true_tags):
            print(f"{t:15} True: {tt:7} Pred: {pt:7}")
        print("-" * 50)


def main():
    (x_train_text, y_train_text), (x_val_text, y_val_text), (x_test_text, y_test_text) = load_data()

    word_vocab = build_vocab(x_train_text)
    tag_map = build_tag_map(y_train_text)

    x_train = encode_and_pad(x_train_text, word_vocab, MAX_LEN)
    x_val = encode_and_pad(x_val_text, word_vocab, MAX_LEN)
    x_test = encode_and_pad(x_test_text, word_vocab, MAX_LEN)

    y_train = encode_and_pad(y_train_text, tag_map, MAX_LEN)
    y_val = encode_and_pad(y_val_text, tag_map, MAX_LEN)
    y_test = encode_and_pad(y_test_text, tag_map, MAX_LEN)

    vocab_size = len(word_vocab)
    tag_count = len(tag_map)

    if os.path.exists(MODEL_PATH):
        print("Loading saved NER model...")
        model = keras.models.load_model(MODEL_PATH)

    else:
        print("Training new NER model...")
        model = build_model(vocab_size, tag_count)
        model.summary()

        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS
        )

        print("Saving model...")
        model.save(MODEL_PATH)

    print("\nRunning test evaluation...")
    evaluate_model(model, x_test, y_test, tag_map)

    show_examples(model, x_test, x_test_text, y_test, tag_map, num_examples=5)


if __name__ == "__main__":
    main()
