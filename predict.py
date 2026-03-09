#!/usr/bin/env python3

"""Advanced edge AI inference pipeline with multiple models and preprocessing."""

import sys
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import pickle
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    model: Any
    vectorizer: Any
    accuracy: float = 0.0

class EdgeAIPipeline:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))

    def load_dataset(self) -> tuple:
        """Simulate loading a larger dataset"""
        print("📊 Loading dataset...")
        # Expanded toy dataset
        texts = [
            "excellent product highly recommend",
            "amazing quality best purchase",
            "great service fast delivery",
            "wonderful experience will buy again",
            "good value for money",
            "terrible product waste of money",
            "horrible quality do not buy",
            "awful service never again",
            "bad experience total disappointment",
            "worst purchase ever"
        ] * 10  # Multiply for more data
        labels = [1] * 50 + [0] * 50
        return texts, labels

    def preprocess_text(self, texts: List[str]) -> np.ndarray:
        """Advanced text preprocessing"""
        print("🔄 Preprocessing text data...")
        # Add some preprocessing steps
        processed = []
        for text in texts:
            # Simple cleaning
            text = text.lower().strip()
            processed.append(text)
        return self.vectorizer.fit_transform(processed) if hasattr(self.vectorizer, 'fit_transform') else self.vectorizer.transform(processed)

    def train_models(self):
        """Train multiple models and select best"""
        print("🤖 Training multiple models...")
        texts, labels = self.load_dataset()
        X = self.preprocess_text(texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'svm': SVC(kernel='linear', random_state=42)
        }

        best_accuracy = 0
        for name, model in models.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            print(f"   {name} accuracy: {accuracy:.3f}")

            config = ModelConfig(name=name, model=model, vectorizer=self.vectorizer, accuracy=accuracy)
            self.models[name] = config

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = config

        print(f"\n🏆 Best model: {self.best_model.name} with {self.best_model.accuracy:.3f} accuracy")
        self.save_models()

    def save_models(self):
        """Save trained models for edge deployment"""
        print("💾 Saving models for edge deployment...")
        for name, config in self.models.items():
            joblib.dump(config, f"{name}_model.joblib")
        print("Models saved.")

    def load_model(self, model_name: str = None):
        """Load model for inference"""
        if model_name is None:
            model_name = self.best_model.name if self.best_model else 'logistic_regression'
        try:
            config = joblib.load(f"{model_name}_model.joblib")
            return config
        except FileNotFoundError:
            print(f"Model {model_name} not found. Run training first.")
            return None

    def predict(self, text: str, model_name: str = None) -> Dict[str, Any]:
        """Run inference with timing and confidence"""
        start_time = time.time()
        config = self.load_model(model_name)
        if not config:
            return {"error": "Model not loaded"}

        # Preprocess input
        processed = self.vectorizer.transform([text.lower().strip()])
        prediction = config.model.predict(processed)[0]
        probabilities = config.model.predict_proba(processed)[0] if hasattr(config.model, 'predict_proba') else None

        inference_time = time.time() - start_time

        result = {
            "input": text,
            "prediction": "positive" if prediction == 1 else "negative",
            "confidence": max(probabilities) if probabilities is not None else None,
            "model": config.name,
            "inference_time_ms": round(inference_time * 1000, 2)
        }

        print(f"🔮 Prediction: {result['prediction']} (confidence: {result['confidence']:.3f}) in {result['inference_time_ms']}ms")
        return result

    def evaluate_model(self, model_name: str = None):
        """Evaluate model performance"""
        config = self.load_model(model_name)
        if not config:
            return

        texts, labels = self.load_dataset()
        X = self.preprocess_text(texts)
        _, X_test, _, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

        predictions = config.model.predict(X_test)
        print(f"\n📈 Evaluation for {config.name}:")
        print(classification_report(y_test, predictions))

def main():
    pipeline = EdgeAIPipeline()

    if len(sys.argv) < 2:
        print("Usage: python predict.py train | predict <text> | evaluate")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        pipeline.train_models()
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: python predict.py predict <text>")
            sys.exit(1)
        text = ' '.join(sys.argv[2:])
        pipeline.predict(text)
    elif command == "evaluate":
        pipeline.evaluate_model()
    else:
        print("Unknown command. Use train, predict, or evaluate.")

if __name__ == "__main__":
    main()
