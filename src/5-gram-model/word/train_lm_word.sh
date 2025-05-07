#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Step 1: Create uppercase word-level text file
echo "Converting grid.txt to word level (uppercase)..."
python3 -c '
with open("../grid.txt", "r") as f_in, open("./grid_word.txt", "w") as f_out:
    for line in f_in:
        # Convert to uppercase without splitting into characters
        upper_line = line.strip().upper()
        f_out.write(upper_line + "\n")
'

# Step 2: Create lexicon file (word to character mapping)
echo "Creating lexicon file..."
python3 -c '
import re

# Get unique words from the training data
words = set()
with open("./grid_word.txt", "r") as f:
    for line in f:
        for word in line.strip().split():
            words.add(word)

# Create lexicon file mapping words to characters
with open("./lexicon.txt", "w") as f_out:
    for word in sorted(words):
        # Each line: WORD W O R D
        characters = " ".join(list(word))
        f_out.write(f"{word} {characters}\n")

print(f"Created lexicon with {len(words)} words")
'

# Step 3: Train word-level language model
echo "Training 5-gram word-level language model..."
lmplz -o 5 -S 4G --discount_fallback < ./grid_word.txt > ./grid_word.arpa

# Step 4: Binarize the model
build_binary grid_word.arpa grid_word.binary

echo "Word-level language model and lexicon created successfully!"