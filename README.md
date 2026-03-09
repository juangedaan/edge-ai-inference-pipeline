# Edge AI Inference Pipeline

A minimal edge inference pipeline: a script takes text and prints a mock prediction.

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
└── infer.py
```

## 🚀 Usage

```bash
python infer.py "some input data"
```

## 📜 License

MIT License
