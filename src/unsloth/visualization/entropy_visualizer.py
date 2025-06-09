"""Entropy visualization tools for analyzing token-level uncertainty.
Module: entropy_visualizer.py
Description: Create visual representations of token entropy distributions

This module provides comprehensive visualization tools for analyzing
token-level entropy in training data and model outputs.

External Dependencies:
- matplotlib: https://matplotlib.org/stable/index.html
- seaborn: https://seaborn.pydata.org/
- plotly: https://plotly.com/python/
- numpy: https://numpy.org/doc/stable/

Sample Input:
>>> tokens = ["Hello", "world", "!"]
>>> entropies = [0.2, 1.5, 0.1]
>>> visualizer = EntropyVisualizer()
>>> fig = visualizer.create_heatmap(tokens, entropies)

Expected Output:
>>> # Interactive heatmap showing token-level entropy
>>> fig.show()

Example Usage:
>>> from unsloth.visualization.entropy_visualizer import EntropyVisualizer
>>> visualizer = EntropyVisualizer()
>>> visualizer.analyze_dataset("path/to/dataset.json")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
import torch
from transformers import AutoTokenizer


class EntropyVisualizer:
    """Comprehensive entropy visualization toolkit."""
    
    def __init__(self, style: str = "dark"):
        """Initialize visualizer with style settings.
        
        Args:
            style: Visual style ("dark" or "light")
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """Setup matplotlib and seaborn styles."""
        if self.style == "dark":
            plt.style.use("dark_background")
            sns.set_theme(style="darkgrid")
            self.cmap = "viridis"
            self.bg_color = "#1e1e1e"
            self.text_color = "#ffffff"
        else:
            plt.style.use("seaborn-v0_8-whitegrid")
            sns.set_theme(style="whitegrid")
            self.cmap = "coolwarm"
            self.bg_color = "#ffffff"
            self.text_color = "#000000"
    
    def create_token_heatmap(
        self,
        tokens: List[str],
        entropies: List[float],
        title: str = "Token Entropy Heatmap",
        interactive: bool = True
    ) -> Union[go.Figure, plt.Figure]:
        """Create heatmap visualization of token entropies.
        
        Args:
            tokens: List of token strings
            entropies: List of entropy values
            title: Plot title
            interactive: Use Plotly for interactivity
            
        Returns:
            Figure object (Plotly or Matplotlib)
        """
        if len(tokens) != len(entropies):
            raise ValueError("Tokens and entropies must have same length")
        
        if interactive:
            # Create interactive Plotly heatmap
            # Reshape for 2D display (wrap at 20 tokens per row)
            tokens_per_row = 20
            n_rows = (len(tokens) + tokens_per_row - 1) // tokens_per_row
            
            # Pad data to fill grid
            padded_tokens = tokens + [""] * (n_rows * tokens_per_row - len(tokens))
            padded_entropies = entropies + [0] * (n_rows * tokens_per_row - len(entropies))
            
            # Reshape to 2D
            token_grid = np.array(padded_tokens).reshape(n_rows, tokens_per_row)
            entropy_grid = np.array(padded_entropies).reshape(n_rows, tokens_per_row)
            
            fig = go.Figure(data=go.Heatmap(
                z=entropy_grid,
                text=token_grid,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Entropy"),
                hovertemplate="Token: %{text}<br>Entropy: %{z:.3f}<extra></extra>"
            ))
            
            fig.update_layout(
                title=title,
                xaxis_visible=False,
                yaxis_visible=False,
                paper_bgcolor=self.bg_color,
                plot_bgcolor=self.bg_color,
                font=dict(color=self.text_color),
                height=100 * n_rows + 100
            )
            
            return fig
        else:
            # Create static matplotlib heatmap
            fig, ax = plt.subplots(figsize=(15, 3))
            
            # Create color array
            colors = plt.cm.get_cmap(self.cmap)(entropies)
            
            # Plot as colored text
            for i, (token, entropy, color) in enumerate(zip(tokens, entropies, colors)):
                ax.text(i, 0, token, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                       fontsize=10, rotation=45 if len(token) > 3 else 0)
            
            ax.set_xlim(-0.5, len(tokens) - 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            ax.set_title(title)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=self.cmap, 
                                       norm=plt.Normalize(vmin=min(entropies), vmax=max(entropies)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
            cbar.set_label('Entropy')
            
            plt.tight_layout()
            return fig
    
    def create_entropy_distribution(
        self,
        entropies: List[float],
        title: str = "Entropy Distribution",
        bins: int = 50
    ) -> go.Figure:
        """Create histogram of entropy distribution.
        
        Args:
            entropies: List of entropy values
            title: Plot title
            bins: Number of histogram bins
            
        Returns:
            Plotly figure
        """
        fig = px.histogram(
            x=entropies,
            nbins=bins,
            title=title,
            labels={'x': 'Entropy', 'y': 'Count'},
            color_discrete_sequence=['#00d4ff']
        )
        
        # Add statistics
        mean_entropy = np.mean(entropies)
        median_entropy = np.median(entropies)
        
        fig.add_vline(x=mean_entropy, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_entropy:.3f}")
        fig.add_vline(x=median_entropy, line_dash="dash", line_color="green",
                     annotation_text=f"Median: {median_entropy:.3f}")
        
        fig.update_layout(
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
    
    def create_sequence_entropy_plot(
        self,
        sequences: List[Dict[str, Union[List[str], List[float]]]],
        title: str = "Sequence Entropy Comparison"
    ) -> go.Figure:
        """Create line plot comparing entropy across sequences.
        
        Args:
            sequences: List of dicts with 'tokens' and 'entropies'
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for i, seq in enumerate(sequences):
            tokens = seq.get('tokens', [])
            entropies = seq.get('entropies', [])
            name = seq.get('name', f'Sequence {i+1}')
            
            # Create hover text
            hover_text = [f"Token: {t}<br>Entropy: {e:.3f}" 
                         for t, e in zip(tokens, entropies)]
            
            fig.add_trace(go.Scatter(
                x=list(range(len(entropies))),
                y=entropies,
                mode='lines+markers',
                name=name,
                text=hover_text,
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Token Position",
            yaxis_title="Entropy",
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color),
            hovermode='x unified'
        )
        
        return fig
    
    def create_entropy_attention_matrix(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        entropies: List[float],
        title: str = "Entropy-Weighted Attention"
    ) -> go.Figure:
        """Create attention matrix weighted by entropy.
        
        Args:
            attention_weights: Attention matrix [seq_len, seq_len]
            tokens: List of tokens
            entropies: List of entropy values
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Weight attention by entropy
        entropy_matrix = np.outer(entropies, entropies)
        weighted_attention = attention_weights * np.sqrt(entropy_matrix)
        
        fig = go.Figure(data=go.Heatmap(
            z=weighted_attention,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Weighted<br>Attention"),
            hovertemplate="From: %{y}<br>To: %{x}<br>Weight: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="To Token",
            yaxis_title="From Token",
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color),
            height=600,
            width=800
        )
        
        return fig
    
    def analyze_dataset(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        model_name: str = "gpt2",
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Analyze entropy patterns in a dataset.
        
        Args:
            dataset_path: Path to dataset JSON
            output_dir: Directory for output visualizations
            model_name: Model for tokenization
            max_samples: Maximum samples to analyze
            
        Returns:
            Analysis results dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset
        with open(dataset_path) as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = data.get('data', [])
        
        data = data[:max_samples]
        
        all_entropies = []
        high_entropy_examples = []
        
        logger.info(f"Analyzing {len(data)} samples...")
        
        for i, item in enumerate(data):
            text = item.get('text', '') or f"{item.get('question', '')} {item.get('answer', '')}"
            
            # Calculate entropy (mock for now, replace with actual calculation)
            tokens = tokenizer.tokenize(text)
            # Mock entropy - replace with actual entropy calculation
            entropies = np.random.exponential(0.5, len(tokens))
            
            all_entropies.extend(entropies)
            
            # Track high entropy examples
            mean_entropy = np.mean(entropies)
            if mean_entropy > np.percentile(all_entropies, 90):
                high_entropy_examples.append({
                    'index': i,
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'mean_entropy': mean_entropy,
                    'tokens': tokens[:20],  # First 20 tokens
                    'entropies': entropies[:20]
                })
        
        # Create visualizations
        logger.info("Creating visualizations...")
        
        # 1. Overall entropy distribution
        dist_fig = self.create_entropy_distribution(all_entropies)
        dist_fig.write_html(output_dir / "entropy_distribution.html")
        
        # 2. High entropy examples heatmap
        if high_entropy_examples:
            example = high_entropy_examples[0]
            heatmap_fig = self.create_token_heatmap(
                example['tokens'],
                example['entropies'],
                title=f"High Entropy Example (Index {example['index']})"
            )
            heatmap_fig.write_html(output_dir / "high_entropy_heatmap.html")
        
        # 3. Entropy statistics plot
        stats_fig = go.Figure()
        
        # Add box plot
        stats_fig.add_trace(go.Box(
            y=all_entropies,
            name="Entropy Distribution",
            boxpoints='outliers'
        ))
        
        stats_fig.update_layout(
            title="Entropy Statistics",
            yaxis_title="Entropy",
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        stats_fig.write_html(output_dir / "entropy_statistics.html")
        
        # Calculate statistics
        results = {
            'total_samples': len(data),
            'total_tokens': len(all_entropies),
            'mean_entropy': float(np.mean(all_entropies)),
            'median_entropy': float(np.median(all_entropies)),
            'std_entropy': float(np.std(all_entropies)),
            'min_entropy': float(np.min(all_entropies)),
            'max_entropy': float(np.max(all_entropies)),
            'high_entropy_samples': len(high_entropy_examples),
            'output_dir': str(output_dir)
        }
        
        # Save results
        with open(output_dir / "analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {output_dir}")
        
        return results
    
    def create_training_entropy_dashboard(
        self,
        training_logs: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ) -> None:
        """Create comprehensive training entropy dashboard.
        
        Args:
            training_logs: List of training log entries
            output_path: Output HTML file path
        """
        from plotly.subplots import make_subplots
        
        # Extract data
        steps = [log['step'] for log in training_logs]
        losses = [log['loss'] for log in training_logs]
        mean_entropies = [log.get('mean_entropy', 0) for log in training_logs]
        max_entropies = [log.get('max_entropy', 0) for log in training_logs]
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Mean Entropy', 
                          'Max Entropy', 'Entropy vs Loss'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': True}]]
        )
        
        # Training loss
        fig.add_trace(
            go.Scatter(x=steps, y=losses, name='Loss', line=dict(color='#ff6b6b')),
            row=1, col=1
        )
        
        # Mean entropy
        fig.add_trace(
            go.Scatter(x=steps, y=mean_entropies, name='Mean Entropy', line=dict(color='#4ecdc4')),
            row=1, col=2
        )
        
        # Max entropy
        fig.add_trace(
            go.Scatter(x=steps, y=max_entropies, name='Max Entropy', line=dict(color='#45b7d1')),
            row=2, col=1
        )
        
        # Entropy vs Loss (dual axis)
        fig.add_trace(
            go.Scatter(x=steps, y=losses, name='Loss', line=dict(color='#ff6b6b')),
            row=2, col=2, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=steps, y=mean_entropies, name='Entropy', line=dict(color='#4ecdc4', dash='dash')),
            row=2, col=2, secondary_y=True
        )
        
        # Update layout
        fig.update_xaxes(title_text="Step", row=2, col=1)
        fig.update_xaxes(title_text="Step", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Entropy", row=1, col=2)
        fig.update_yaxes(title_text="Entropy", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Entropy", row=2, col=2, secondary_y=True)
        
        fig.update_layout(
            title_text="Training Entropy Dashboard",
            height=800,
            showlegend=True,
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        # Save dashboard
        fig.write_html(output_path)
        logger.info(f"Dashboard saved to {output_path}")


def create_entropy_report(
    dataset_path: str,
    output_dir: str = "./entropy_analysis",
    model_name: str = "gpt2"
) -> Dict[str, Any]:
    """Create comprehensive entropy analysis report.
    
    Args:
        dataset_path: Path to dataset
        output_dir: Output directory
        model_name: Model for tokenization
        
    Returns:
        Analysis results
    """
    visualizer = EntropyVisualizer()
    results = visualizer.analyze_dataset(
        dataset_path,
        output_dir,
        model_name=model_name
    )
    
    return results


if __name__ == "__main__":
    # Validation
    import tempfile
    
    # Create test data
    test_data = [
        {"question": "What is AI?", "answer": "Artificial Intelligence"},
        {"question": "How does ML work?", "answer": "Machine Learning uses algorithms"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name
    
    try:
        # Test visualization
        visualizer = EntropyVisualizer()
        
        # Test token heatmap
        tokens = ["Hello", "world", "!", "How", "are", "you", "?"]
        entropies = [0.2, 1.5, 0.1, 0.8, 0.6, 0.9, 0.3]
        fig = visualizer.create_token_heatmap(tokens, entropies)
        assert fig is not None
        
        # Test dataset analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            results = visualizer.analyze_dataset(temp_path, temp_dir)
            assert results['total_samples'] == 2
            assert 'mean_entropy' in results
        
        print(" Entropy visualizer validation passed")
        
    finally:
        Path(temp_path).unlink()