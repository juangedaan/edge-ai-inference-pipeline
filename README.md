# Edge AI Inference Pipeline

A comprehensive edge AI pipeline for text sentiment analysis. Trains multiple ML models (Logistic Regression, Random Forest, SVM), evaluates performance, and provides optimized inference with timing metrics.

```mermaid
flowchart LR
    Data[Text Data] --> Preprocess[TF-IDF Vectorization]
    Preprocess --> Train[Train LR / RF / SVM]
    Train --> Evaluate[Model Evaluation]
    Evaluate --> SelectBest[Select Best Model]
    SelectBest --> SaveModel[Save Models]
    SaveModel --> Predict[Predict with Latency Metrics]
```

## 📂 Structure

```
edge-ai-inference-pipeline/
├── README.md
├── requirements.txt
├── predict.py  # Full ML pipeline with training, evaluation, inference
├── *_model.joblib  # Saved models (created after training)
```

## 🚀 Usage

```bash
# Train models
python predict.py train

# Run inference
python predict.py predict "great product amazing quality"

# Evaluate performance
python predict.py evaluate
```

## 🏗️ Pipeline Features

- **Multi-Model Training**: Logistic Regression, Random Forest, SVM
- **Advanced Preprocessing**: TF-IDF vectorization with n-grams
- **Model Selection**: Automatic best model selection
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Edge Optimization**: Fast inference with timing measurements
- **Model Persistence**: Save/load models for deployment

## 📜 License

MIT License
