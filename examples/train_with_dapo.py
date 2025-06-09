"""Production example: Train model with DAPO RL for superior reasoning.
Module: train_with_dapo.py
Description: Complete example of training with DAPO reinforcement learning

This example demonstrates how to use the DAPO algorithm to train models
with better reasoning capabilities through decoupled clipping and dynamic sampling.

External Dependencies:
- transformers: https://huggingface.co/docs/transformers/
- torch: https://pytorch.org/docs/stable/
- datasets: https://huggingface.co/docs/datasets/

Sample Input:
>>> python train_with_dapo.py --model unsloth/Phi-3.5-mini-instruct --dataset ./data/qa.json

Expected Output:
>>> Training with DAPO...
>>> Step 100: Loss=0.234, Entropy=1.45, KL=0.02
>>> Model saved to ./outputs/dapo_model

Example Usage:
>>> # Basic usage
>>> python train_with_dapo.py --model unsloth/Phi-3.5-mini-instruct
>>> # With custom config
>>> python train_with_dapo.py --model meta-llama/Llama-3.2-1B --clip-upper 1.3 --dynamic-sampling
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from unsloth.training.dapo_rl import DAPOConfig, DAPOTrainer, create_dapo_trainer
from unsloth.data.thinking_enhancer import ThinkingEnhancer, StudentTeacherConfig
from unsloth.visualization.entropy_visualizer import EntropyVisualizer


def load_qa_dataset(path: str, max_samples: int = None) -> List[Dict]:
    """Load Q&A dataset from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = data.get('data', data.get('samples', []))
    
    if max_samples:
        data = data[:max_samples]
    
    return data


def prepare_dapo_batch(
    data: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """Prepare batch for DAPO training."""
    texts = []
    for item in data:
        if 'text' in item:
            texts.append(item['text'])
        else:
            # Format Q&A as conversation
            q = item.get('question', '')
            a = item.get('answer', '')
            text = f"Human: {q}\n\nAssistant: {a}"
            texts.append(text)
    
    # Tokenize
    batch = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Add mock rewards for demonstration (replace with actual reward model)
    batch['rewards'] = torch.randn(len(texts), max_length) * 0.1
    
    return batch


async def enhance_with_thinking(
    data: List[Dict],
    config: StudentTeacherConfig
) -> List[Dict]:
    """Enhance dataset with student-teacher thinking."""
    enhancer = ThinkingEnhancer(config)
    enhanced_data = []
    
    logger.info(f"Enhancing {len(data)} samples with thinking...")
    
    for item in data:
        q = item.get('question', '')
        a = item.get('answer', '')
        
        enhanced = await enhancer.enhance_qa_pair(q, a)
        enhanced_data.append({
            'question': enhanced['question'],
            'answer': enhanced['final_answer'],
            'thinking': enhanced.get('student_thinking', ''),
            'teacher_feedback': enhanced.get('teacher_feedback', ''),
            'original_answer': a
        })
    
    return enhanced_data


def main():
    parser = argparse.ArgumentParser(description="Train with DAPO RL")
    parser.add_argument("--model", type=str, default="unsloth/Phi-3.5-mini-instruct",
                       help="Model to fine-tune")
    parser.add_argument("--dataset", type=str, default="./data/qa_dataset.json",
                       help="Path to Q&A dataset")
    parser.add_argument("--output", type=str, default="./outputs/dapo_model",
                       help="Output directory")
    parser.add_argument("--enhance", action="store_true",
                       help="Enhance dataset with thinking first")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to use")
    parser.add_argument("--clip-lower", type=float, default=0.8,
                       help="DAPO lower clip bound")
    parser.add_argument("--clip-upper", type=float, default=1.28,
                       help="DAPO upper clip bound")
    parser.add_argument("--dynamic-sampling", action="store_true",
                       help="Enable dynamic sampling")
    parser.add_argument("--visualize", action="store_true",
                       help="Create entropy visualizations")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model: {args.model}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Create reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    data = load_qa_dataset(args.dataset, args.max_samples)
    
    # Enhance with thinking if requested
    if args.enhance:
        config = StudentTeacherConfig(
            student_model="claude-3-haiku-20240307",
            teacher_model="claude-3-5-sonnet-20241022"
        )
        data = asyncio.run(enhance_with_thinking(data, config))
        
        # Save enhanced dataset
        enhanced_path = output_dir / "enhanced_dataset.json"
        with open(enhanced_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Enhanced dataset saved to {enhanced_path}")
    
    # Create DAPO configuration
    dapo_config = DAPOConfig(
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        dynamic_sampling=args.dynamic_sampling,
        token_level_pg=True,
        gradient_accumulation_steps=4
    )
    
    # Create trainer
    trainer = DAPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=dapo_config
    )
    
    # Training loop
    logger.info("Starting DAPO training...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    training_logs = []
    global_step = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Process in batches
        for i in range(0, len(data), args.batch_size):
            batch_data = data[i:i + args.batch_size]
            batch = prepare_dapo_batch(batch_data, tokenizer)
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            metrics = trainer.train_step(batch)
            
            # Backward pass
            loss = metrics['loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), dapo_config.max_grad_norm)
            
            # Optimizer step
            if (global_step + 1) % dapo_config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # Log metrics
            if global_step % 10 == 0:
                logger.info(
                    f"Step {global_step}: Loss={metrics['loss']:.4f}, "
                    f"PG Loss={metrics['pg_loss']:.4f}, "
                    f"Entropy={metrics['entropy']:.4f}, "
                    f"KL={metrics['approx_kl']:.4f}"
                )
                
                training_logs.append({
                    'step': global_step,
                    'loss': metrics['loss'],
                    'pg_loss': metrics['pg_loss'],
                    'entropy': metrics['entropy'],
                    'kl': metrics['approx_kl']
                })
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training logs
    logs_path = output_dir / "training_logs.json"
    with open(logs_path, 'w') as f:
        json.dump(training_logs, f, indent=2)
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("Creating entropy visualizations...")
        visualizer = EntropyVisualizer()
        
        # Analyze final model entropy
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Sample entropy analysis
        sample_text = data[0].get('text', f"{data[0]['question']} {data[0]['answer']}")
        tokens = tokenizer.tokenize(sample_text)[:50]
        
        # Mock entropy values (replace with actual calculation)
        import numpy as np
        entropies = np.random.exponential(0.5, len(tokens))
        
        fig = visualizer.create_token_heatmap(tokens, entropies.tolist())
        fig.write_html(vis_dir / "sample_entropy.html")
        
        # Training dashboard
        visualizer.create_training_entropy_dashboard(
            training_logs,
            vis_dir / "training_dashboard.html"
        )
        
        logger.info(f"Visualizations saved to {vis_dir}")
    
    logger.info(" DAPO training complete!")
    
    # Print example generation
    logger.info("\nExample generation:")
    model.eval()
    
    test_prompt = "Human: What is machine learning?\n\nAssistant:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.9,
            do_sample=True,
            top_p=0.95
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Response: {response}")


if __name__ == "__main__":
    # For testing with mock data
    import sys
    if "--test" in sys.argv:
        # Create test dataset
        test_data = [
            {"question": "What is AI?", "answer": "Artificial Intelligence is the simulation of human intelligence by machines."},
            {"question": "Explain neural networks", "answer": "Neural networks are computing systems inspired by biological neural networks."},
            {"question": "What is deep learning?", "answer": "Deep learning is a subset of machine learning using multi-layer neural networks."}
        ]
        
        test_path = Path("./test_data/qa_test.json")
        test_path.parent.mkdir(exist_ok=True)
        
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f" Test data created at {test_path}")
        logger.info("Run: python train_with_dapo.py --dataset ./test_data/qa_test.json --max-samples 3")
    else:
        main()