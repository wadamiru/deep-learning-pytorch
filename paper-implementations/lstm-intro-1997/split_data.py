"""
===============================================================================
split_data.py

Purpose:
    Splits a raw text file into training, validation, and test sets for 
    character-level LSTM experiments. This is useful for reproducible 
    experiments.

Usage:
    1. Place your raw text file (full.txt) in the `data/` folder.
    2. Adjust INPUT_PATH if needed.
    3. Run:
        python split_data.py
    4. The script will create:
        - data/train.txt
        - data/val.txt
        - data/test.txt

Notes:
    - The split is done sequentially by character, not randomly. This preserves 
      temporal dependencies for sequence modeling.
    - Default split ratios:
        - Train: 80%
        - Validation: 10%
        - Test: 10%
    - You can adjust the ratios by changing train_ratio, val_ratio, test_ratio.
    - Ensure the `data/` folder exists before running.
===============================================================================
"""

import os

INPUT_PATH = "data/full.txt"
TRAIN_PATH = "data/train.txt"
VAL_PATH = "data/val.txt"
TEST_PATH = "data/test.txt"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

n = len(text)
train_end = int(n * train_ratio)
val_end = train_end + int(n * val_ratio)

train_text = text[:train_end]
val_text = text[train_end:val_end]
test_text = text[val_end:]

os.makedirs("data", exist_ok=True)

with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    f.write(train_text)

with open(VAL_PATH, "w", encoding="utf-8") as f:
    f.write(val_text)

with open(TEST_PATH, "w", encoding="utf-8") as f:
    f.write(test_text)

print("Done splitting:")
print(f"Train: {len(train_text)} chars")
print(f"Val:   {len(val_text)} chars")
print(f"Test:  {len(test_text)} chars")
