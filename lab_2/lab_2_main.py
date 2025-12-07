import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

MODEL_PATH = "reuters_conv1d_model.keras"
NUM_WORDS = 10000
MAXLEN = 200
EMBED_DIM = 128
BATCH_SIZE = 128
EPOCHS = 8


def load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words=NUM_WORDS, test_split=0.2
    )
    num_classes = np.max(y_train) + 1

    x_train = pad_sequences(x_train, maxlen=MAXLEN)
    x_test = pad_sequences(x_test, maxlen=MAXLEN)

    print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")
    print(f"Number of classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), num_classes


def build_model(num_classes):
    model = keras.Sequential([
        layers.Embedding(NUM_WORDS, EMBED_DIM, input_length=MAXLEN),
        layers.Conv1D(128, kernel_size=5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(1e-3),
        metrics=["accuracy"]
    )
    return model


def plot_history(history):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def show_confusion_matrix(cm):
    plt.figure(figsize=(9,8))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def decode_newswire(sequence, reverse_word_index):
    return " ".join(
        [reverse_word_index.get(i - 3, "?") for i in sequence if i >= 3]
    )


def evaluate_model(model, x_test, y_test):
    print("\nEvaluating model on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_proba = model.predict(x_test)
    y_pred = np.argmax(y_proba, axis=1)

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    pd.set_option("display.max_rows", 200)
    print("\nClassification Report:")
    print(report_df)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    show_confusion_matrix(cm)

    word_index = reuters.get_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(x_test), size=8, replace=False)

    print("\nRandom Test Predictions:\n")
    for idx in indices:
        text = decode_newswire(x_test[idx], reverse_word_index)
        true = int(y_test[idx])
        pred = int(y_pred[idx])
        prob = float(np.max(y_proba[idx]))

        snippet = " ".join(text.split()[:80])
        print(f"True: {true} → Predicted: {pred} | Prob: {prob:.3f}")
        print(snippet)
        print("-" * 80)


def main():
    (x_train, y_train), (x_test, y_test), num_classes = load_and_preprocess()

    if os.path.exists(MODEL_PATH):
        print(f"\nFound saved model: {MODEL_PATH}")
        print("Loading model and running test evaluation only...\n")
        model = keras.models.load_model(MODEL_PATH)

    else:
        print("\nNo saved model found. Training new model...\n")
        model = build_model(num_classes)
        model.summary()

        history = model.fit(
            x_train, y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2
        )

        plot_history(history)

        print(f"\nSaving model to: {MODEL_PATH}")
        model.save(MODEL_PATH)

    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
