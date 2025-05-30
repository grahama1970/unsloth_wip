# Unsloth Fine-tuning Commands

This directory contains slash commands for the Unsloth fine-tuning module.

## Available Commands

### Data Preparation
-  - Prepare and format datasets for fine-tuning
-  - Import Q&A tuples from other modules

### Model Configuration
-  - Configure LoRA adapter parameters

### Training
-  - Start model fine-tuning with Unsloth

### Export & Validation
-  - Export trained LoRA adapters
-  - Validate fine-tuned model performance

### Module Communication
-  - Register with Module Communicator
-  - Receive data from other modules

### Getting Started
-  - Quick guide and common workflows

## Integration with Other Modules

This module integrates with:
- **ArangoDB**: Receives Q&A tuples for training
- **SPARTA**: Receives processed cybersecurity data
- **Claude Max Proxy**: Can receive enhanced Q&A data
- **Marker**: Can process extracted document conversations

## Example Workflow



---
*Updated: May 28, 2025 - Replaced incorrect extraction commands with proper fine-tuning commands*
EOF'