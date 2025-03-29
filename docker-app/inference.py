
import sys
import json

def mock_infer(text):
    if "great" in text:
        return {"sentiment": "positive", "score": 0.92}
    else:
        return {"sentiment": "neutral", "score": 0.5}

if __name__ == "__main__":
    input_text = " ".join(sys.argv[1:]) or "default text"
    result = mock_infer(input_text)
    print(json.dumps(result, indent=2))
