#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Convert to character level WITH UPPERCASE
echo "Converting grid.txt to character level (uppercase)..."
python3 -c '
with open("../grid.txt", "r") as f_in, open("./grid_char.txt", "w") as f_out:
    for line in f_in:
        # Convert to uppercase
        upper_line = line.strip().upper()
        f_out.write(" ".join(list(upper_line)) + "\n")
'

# Train language model
echo "Training 5-gram language model..."
lmplz -o 5 -S 4G --discount_fallback < ./grid_char.txt > ./grid_char.arpa

# Binarize the model
build_binary grid_char.arpa grid_char.binary

echo "Language model created successfully!"