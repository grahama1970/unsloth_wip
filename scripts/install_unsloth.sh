#!/bin/bash
# Install unsloth after other dependencies

echo "Installing unsloth from GitHub..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
echo "Unsloth installation complete!"