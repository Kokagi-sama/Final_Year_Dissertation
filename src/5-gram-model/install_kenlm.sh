#!/bin/bash

set -e  # Exit immediately if a command fails

# Define installation directory
INSTALL_DIR="/opt/kenlm"

# Update system and install dependencies
echo "Installing required dependencies..."
sudo apt update && sudo apt install -y build-essential cmake libboost-all-dev git

# Clone KenLM repository
echo "Cloning KenLM repository into $INSTALL_DIR..."
sudo rm -rf "$INSTALL_DIR"  # Remove previous installation if exists
sudo git clone https://github.com/kpu/kenlm.git "$INSTALL_DIR"

# Build KenLM
echo "Building KenLM..."
cd "$INSTALL_DIR"
sudo mkdir -p build && cd build
sudo cmake ..
sudo make -j$(nproc)  # Use all available CPU cores for faster compilation

# Install KenLM binaries system-wide
echo "Installing KenLM binaries..."
sudo ln -sf "$INSTALL_DIR/build/bin/lmplz" /usr/local/bin/lmplz
sudo ln -sf "$INSTALL_DIR/build/bin/build_binary" /usr/local/bin/build_binary

# Verify installation
echo "Verifying installation..."
if command -v lmplz &> /dev/null && command -v build_binary &> /dev/null; then
    echo "KenLM installed successfully!"
else
    echo "KenLM installation failed!"
    exit 1
fi

# Test KenLM binaries
echo "Testing KenLM binaries..."
lmplz --help
build_binary --help

echo "KenLM is ready to use!"