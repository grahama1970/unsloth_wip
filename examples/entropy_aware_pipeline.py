"""Production example: Complete entropy-aware training pipeline.
Module: entropy_aware_pipeline.py  
Description: End-to-end pipeline with entropy-aware training and visualization

This example shows how to use all entropy-aware features together for
optimal training results with comprehensive analysis.

External Dependencies:
- torch: https://pytorch.org/docs/stable/
- transformers: https://huggingface.co/docs/transformers/
- datasets: https://huggingface.co/docs/datasets/

Sample Input:
>>> python entropy_aware_pipeline.py --model unsloth/Phi-3.5-mini-instruct --dataset ./data/qa.json

Expected Output:
>>> Analyzing dataset entropy...
>>> High entropy tokens: 23.5%
>>> Training with entropy weighting...
>>> Final perplexity: 12.3

Example Usage:
>>> # Basic usage
>>> python entropy_aware_pipeline.py --dataset ./data/qa.json
>>> # With RunPod
>>> python entropy_aware_pipeline.py --dataset ./data/qa.json --use-runpod --gpu-type A100_80GB
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger

from unsloth.core.enhanced_config import EnhancedTrainingConfig
from unsloth.data.entropy_aware_thinking_enhancer import EntropyAwareThinkingEnhancer
from unsloth.data.msmarco_loader import MSMARCODataLoader, MSMARCOConfig
from unsloth.evaluation.entropy_evaluator import EntropyEvaluator
from unsloth.pipeline.complete_training_pipeline import CompletePipeline
from unsloth.training.entropy_aware_trainer import EntropyAwareTrainer, EntropyAwareTrainingConfig
from unsloth.training.entropy_utils import calculate_token_entropy
from unsloth.visualization.entropy_visualizer import EntropyVisualizer, create_entropy_report
from unsloth.upload.entropy_aware_hub_uploader import EntropyAwareHubUploader


class EntropyAwarePipeline:
    """Complete entropy-aware training pipeline."""
    
    def __init__(
        self,
        model_name: str = "unsloth/Phi-3.5-mini-instruct",
        dataset_path: str = None,
        output_dir: str = "./outputs/entropy_pipeline",
        use_runpod: bool = False,
        gpu_type: str = "RTX_4090"
    ):
        """Initialize entropy-aware pipeline.
        
        Args:
            model_name: Model to fine-tune
            dataset_path: Path to dataset
            output_dir: Output directory
            use_runpod: Use RunPod for training
            gpu_type: GPU type for RunPod
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.use_runpod = use_runpod
        self.gpu_type = gpu_type
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.visualizer = EntropyVisualizer()
        self.config = self._create_config()
        
    def _create_config(self) -> EnhancedTrainingConfig:
        """Create enhanced training configuration."""
        config = EnhancedTrainingConfig(
            model_name=self.model_name,
            dataset_path=self.dataset_path,
            output_dir=str(self.output_dir),
            # Entropy-aware settings
            entropy_aware_enabled=True,
            entropy_threshold=0.5,
            entropy_weight_scale=2.0,
            use_entropy_sampling=True,
            # Training settings
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            learning_rate=2e-4,
            # LoRA settings
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            # RunPod settings
            use_runpod=self.use_runpod,
            runpod_gpu_type=self.gpu_type
        )
        
        return config
    
    async def analyze_dataset_entropy(self) -> Dict:
        """Analyze entropy patterns in dataset."""
        logger.info("Analyzing dataset entropy patterns...")
        
        # Create entropy report
        results = create_entropy_report(
            self.dataset_path,
            str(self.vis_dir / "dataset_analysis"),
            model_name=self.model_name
        )
        
        # Log statistics
        logger.info(f"Dataset entropy statistics:")
        logger.info(f"  Mean entropy: {results['mean_entropy']:.3f}")
        logger.info(f"  Std entropy: {results['std_entropy']:.3f}")
        logger.info(f"  High entropy samples: {results['high_entropy_samples']}")
        
        return results
    
    async def enhance_with_entropy_awareness(self, data: List[Dict]) -> List[Dict]:
        """Enhance dataset with entropy-aware thinking."""
        logger.info("Enhancing dataset with entropy-aware thinking...")
        
        enhancer = EntropyAwareThinkingEnhancer(
            student_model="claude-3-haiku-20240307",
            teacher_model="claude-3-5-sonnet-20241022",
            entropy_threshold=0.5
        )
        
        enhanced_data = []
        for item in data[:10]:  # Limit for example
            q = item.get('question', '')
            a = item.get('answer', '')
            
            enhanced = await enhancer.enhance_with_entropy(q, a)
            enhanced_data.append(enhanced)
        
        # Visualize enhancement results
        self._visualize_enhancement(enhanced_data)
        
        return enhanced_data
    
    def _visualize_enhancement(self, enhanced_data: List[Dict]):
        """Create visualizations for enhancement results."""
        if not enhanced_data:
            return
        
        # Extract entropy data
        all_tokens = []
        all_entropies = []
        
        for item in enhanced_data[:5]:  # First 5 examples
            if 'token_entropies' in item:
                tokens = item['tokens'][:30]  # Limit tokens
                entropies = item['token_entropies'][:30]
                
                all_tokens.extend(tokens)
                all_entropies.extend(entropies)
                
                # Create individual heatmap
                fig = self.visualizer.create_token_heatmap(
                    tokens, entropies,
                    title=f"Enhanced Example: {item.get('question', '')[:50]}..."
                )
                fig.write_html(self.vis_dir / f"enhanced_example_{len(all_tokens)}.html")
        
        # Create overall distribution
        if all_entropies:
            dist_fig = self.visualizer.create_entropy_distribution(
                all_entropies,
                title="Enhanced Dataset Entropy Distribution"
            )
            dist_fig.write_html(self.vis_dir / "enhanced_distribution.html")
    
    async def train_with_entropy_weighting(self) -> str:
        """Train model with entropy-aware loss weighting."""
        logger.info("Starting entropy-aware training...")
        
        # Create entropy-aware training config
        entropy_config = EntropyAwareTrainingConfig(
            base_config=self.config,
            entropy_weight_scale=2.0,
            high_entropy_threshold=0.7,
            low_entropy_threshold=0.3,
            use_dynamic_weighting=True
        )
        
        # Initialize trainer
        trainer = EntropyAwareTrainer(
            model_name=self.model_name,
            dataset_path=self.dataset_path,
            config=entropy_config,
            output_dir=str(self.output_dir)
        )
        
        # Train model
        if self.use_runpod:
            from unsloth.runpod.training_ops import TrainingOperations
            
            ops = TrainingOperations()
            result = await ops.run_training(
                trainer=trainer,
                gpu_type=self.gpu_type,
                config=entropy_config
            )
            adapter_path = result['adapter_path']
        else:
            result = trainer.train()
            adapter_path = str(self.output_dir / "adapter")
        
        # Save training metrics
        self._save_training_metrics(result)
        
        return adapter_path
    
    def _save_training_metrics(self, result: Dict):
        """Save and visualize training metrics."""
        metrics_path = self.output_dir / "training_metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Create training dashboard if logs available
        if 'training_logs' in result:
            self.visualizer.create_training_entropy_dashboard(
                result['training_logs'],
                self.vis_dir / "training_dashboard.html"
            )
    
    async def evaluate_with_entropy_metrics(self, adapter_path: str) -> Dict:
        """Evaluate model with entropy-specific metrics."""
        logger.info("Evaluating model with entropy metrics...")
        
        evaluator = EntropyEvaluator(
            model_path=adapter_path,
            base_model=self.model_name
        )
        
        # Load test data
        test_data = []
        if self.dataset_path:
            with open(self.dataset_path) as f:
                data = json.load(f)
                test_data = data[-100:]  # Last 100 samples for test
        
        # Evaluate
        results = await evaluator.evaluate(
            test_data=test_data,
            calculate_perplexity=True,
            analyze_entropy_correlation=True
        )
        
        # Visualize results
        self._visualize_evaluation(results)
        
        return results
    
    def _visualize_evaluation(self, results: Dict):
        """Create evaluation visualizations."""
        import plotly.graph_objects as go
        
        # Entropy vs Perplexity scatter plot
        if 'sample_results' in results:
            entropies = [r['mean_entropy'] for r in results['sample_results']]
            perplexities = [r['perplexity'] for r in results['sample_results']]
            
            fig = go.Figure(data=go.Scatter(
                x=entropies,
                y=perplexities,
                mode='markers',
                marker=dict(
                    size=8,
                    color=entropies,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Entropy")
                ),
                text=[f"Sample {i}" for i in range(len(entropies))],
                hovertemplate="Entropy: %{x:.3f}<br>Perplexity: %{y:.2f}<extra></extra>"
            ))
            
            fig.update_layout(
                title="Entropy vs Perplexity Correlation",
                xaxis_title="Mean Token Entropy",
                yaxis_title="Perplexity",
                paper_bgcolor=self.visualizer.bg_color,
                plot_bgcolor=self.visualizer.bg_color,
                font=dict(color=self.visualizer.text_color)
            )
            
            fig.write_html(self.vis_dir / "entropy_perplexity_correlation.html")
    
    async def upload_with_entropy_metadata(self, adapter_path: str, hub_id: str) -> str:
        """Upload model with entropy training metadata."""
        logger.info(f"Uploading to HuggingFace Hub: {hub_id}")
        
        uploader = EntropyAwareHubUploader()
        
        # Prepare metadata
        metadata = {
            "training_type": "entropy_aware",
            "entropy_config": {
                "weight_scale": self.config.entropy_weight_scale,
                "threshold": self.config.entropy_threshold,
                "dynamic_weighting": True
            },
            "pipeline_version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        # Upload
        result = await uploader.upload_with_metadata(
            adapter_path=adapter_path,
            hub_id=hub_id,
            metadata=metadata,
            include_visualizations=True,
            visualization_dir=str(self.vis_dir)
        )
        
        return result['url']
    
    async def run_complete_pipeline(self) -> Dict:
        """Run the complete entropy-aware pipeline."""
        logger.info(" Starting Entropy-Aware Training Pipeline")
        
        results = {
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Analyze dataset entropy
            entropy_analysis = await self.analyze_dataset_entropy()
            results["entropy_analysis"] = entropy_analysis
            
            # Step 2: Load and enhance dataset
            with open(self.dataset_path) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = data.get('data', [])
            
            enhanced_data = await self.enhance_with_entropy_awareness(data)
            results["enhanced_samples"] = len(enhanced_data)
            
            # Step 3: Train with entropy weighting
            adapter_path = await self.train_with_entropy_weighting()
            results["adapter_path"] = adapter_path
            
            # Step 4: Evaluate with entropy metrics
            eval_results = await self.evaluate_with_entropy_metrics(adapter_path)
            results["evaluation"] = eval_results
            
            # Step 5: Upload if hub ID provided
            if hasattr(self, 'hub_id') and self.hub_id:
                hub_url = await self.upload_with_entropy_metadata(adapter_path, self.hub_id)
                results["hub_url"] = hub_url
            
            results["status"] = "completed"
            logger.info(" Entropy-Aware Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        # Save final results
        results_path = self.output_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Entropy-Aware Training Pipeline")
    parser.add_argument("--model", type=str, default="unsloth/Phi-3.5-mini-instruct",
                       help="Model to fine-tune")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--output", type=str, default="./outputs/entropy_pipeline",
                       help="Output directory")
    parser.add_argument("--use-runpod", action="store_true",
                       help="Use RunPod for training")
    parser.add_argument("--gpu-type", type=str, default="RTX_4090",
                       help="GPU type for RunPod")
    parser.add_argument("--hub-id", type=str, default=None,
                       help="HuggingFace Hub ID for upload")
    parser.add_argument("--test", action="store_true",
                       help="Run with test data")
    
    args = parser.parse_args()
    
    # Create test data if requested
    if args.test:
        test_dir = Path("./test_data")
        test_dir.mkdir(exist_ok=True)
        
        test_data = [
            {
                "question": "What is entropy in information theory?",
                "answer": "Entropy measures the average information content or uncertainty in a message."
            },
            {
                "question": "How does machine learning work?",
                "answer": "Machine learning algorithms learn patterns from data to make predictions."
            },
            {
                "question": "Explain neural networks",
                "answer": "Neural networks are computational models inspired by biological neurons."
            }
        ]
        
        test_path = test_dir / "entropy_test.json"
        with open(test_path, 'w') as f:
            json.dump({"data": test_data}, f, indent=2)
        
        logger.info(f" Test data created at {test_path}")
        args.dataset = str(test_path)
    
    # Run pipeline
    pipeline = EntropyAwarePipeline(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        use_runpod=args.use_runpod,
        gpu_type=args.gpu_type
    )
    
    if args.hub_id:
        pipeline.hub_id = args.hub_id
    
    # Run async pipeline
    results = asyncio.run(pipeline.run_complete_pipeline())
    
    # Print summary
    logger.info("\n Pipeline Summary:")
    logger.info(f"Status: {results['status']}")
    if 'entropy_analysis' in results:
        logger.info(f"Mean dataset entropy: {results['entropy_analysis']['mean_entropy']:.3f}")
    if 'evaluation' in results:
        logger.info(f"Final perplexity: {results['evaluation'].get('perplexity', 'N/A')}")
    if 'hub_url' in results:
        logger.info(f"Model uploaded to: {results['hub_url']}")


if __name__ == "__main__":
    main()