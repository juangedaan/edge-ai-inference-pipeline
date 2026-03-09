#!/usr/bin/env python3

"""Train a simple model and run inference."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import sys

MODEL_PATH = "model.joblib"


def train():
    # small toy dataset
    texts = ["good product", "great service", "bad quality", "terrible experience"]
    labels = [1, 1, 0, 0]
    vect = CountVectorizer()
    X = vect.fit_transform(texts)
    clf = LogisticRegression().fit(X, labels)
    joblib.dump((vect, clf), MODEL_PATH)
    print("Model trained and saved")


def predict(text):
    vect, clf = joblib.load(MODEL_PATH)
    X = vect.transform([text])
    pred = clf.predict(X)[0]
    print(f"Input: {text} -> prediction: {pred}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <text>")
        sys.exit(1)
    if sys.argv[1] == "train":
        train()
    else:
        predict(' '.join(sys.argv[1:]))
