import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

DATA_DIR = "data/static"
MODEL_PATH = "model/asl_model.pkl"

def load_data(data_dir):
    X, y = [], []
    labels = []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".csv"):
            label = file.replace(".csv", "")
            df = pd.read_csv(f"{data_dir}/{file}", header=None)
            for _, row in df.iterrows():
                X.append(row.values)
                y.append(label)
            labels.append(label)
    return np.array(X), np.array(y), labels

def train():
    print("Loading data...")
    X, y, labels = load_data(DATA_DIR)
    print(f"Loaded {len(X)} samples across {len(labels)} classes: {labels}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "labels": labels}, f)
    print(f"\n✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()