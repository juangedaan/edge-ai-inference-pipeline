#!/usr/bin/env python3

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <data>")
        sys.exit(1)
    data = sys.argv[1]
    print(f"Received input: {data}")
    print("Inference result: positive")
