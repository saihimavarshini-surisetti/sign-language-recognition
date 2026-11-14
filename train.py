"""Train the sign classifier from data/landmarks.csv."""
import os, json
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


def augment(X, n=2, noise=0.01):
    """Cheap augmentation: add small Gaussian noise to landmarks."""
    out = [X]
    for _ in range(n):
        out.append(X + np.random.normal(0, noise, X.shape))
    return np.vstack(out)


def main():
    path = os.path.join("data", "landmarks.csv")
    if not os.path.exists(path):
        raise SystemExit("data/landmarks.csv not found. Run collect_data.py first.")
    df = pd.read_csv(path)
    X = df.drop(columns="label").values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"{len(df)} samples, {len(le.classes_)} classes: {list(le.classes_)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_aug = augment(X_tr)
    y_aug = np.tile(y_tr, 3)

    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    model.fit(X_aug, y_aug)

    pred = model.predict(X_te)
    print(f"\nTest accuracy: {accuracy_score(y_te, pred):.4f}\n")
    print(classification_report(y_te, pred, target_names=le.classes_))

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump({"model": model, "classes": list(le.classes_)}, "models/sign_model.pkl")

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, display_labels=le.classes_, ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=120)
    print("Saved models/sign_model.pkl and outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
