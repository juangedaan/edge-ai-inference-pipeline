# Edge AI Inference Pipeline

An edge inference demo: trains a tiny text classifier and runs predictions with scikit-learn.

```mermaid
flowchart LR
    Input[Data] --> Script[infer.py]
    Script --> Output[Prediction]
```

## 📂 Structure

```
edge-ai-inference-pipeline/
├── README.md
├── requirements.txt
├── infer.py (old stub)
└── predict.py  # new training/inference script
```

## 🚀 Usage

```bash
python infer.py "some input data"
```

## 📜 License

MIT License
